import os
import errno
import pathlib
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pysmiles import read_smiles
from IPython.display import clear_output
from openbabel import openbabel
from Bio import SeqIO
from periodictable import elements
import re

ELEMENTS_DICT = {el.symbol: el.number for el in elements}
ELEMENTS_SET = set([el.symbol for el in elements])


def convert_fasta_to_smiles(input_fasta_file_path, output_smiles_file_path):
    """
    convert_fasta_to_smiles converts a FASTA file located at
    input_fasta_file_path and outputs a smiles file at the location
    output_smiles_file_path
    Args:
        input_fasta_file_path (os.path): The path to the input FASTA file.
        output_smiles_file_path (os.path): The path to the output smiles file.
    Returns:
        list: This list contains smiles strings as elements. These strings are
        the result of the conversion of sequences in the FASTA file.
    """

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats('fasta', 'smi')

    mol = openbabel.OBMol()
    smiles_list = []

    with open(input_fasta_file_path, 'r') as input_file, \
            open(output_smiles_file_path, 'a') as output_file:
        for i, fasta_string in enumerate(SeqIO.parse(input_file, 'fasta')):
            obConversion.ReadString(mol, str(fasta_string.seq))
            output_smiles_string = obConversion.WriteString(mol)
            for char in ['[', ']', '.']:
                output_smiles_string = output_smiles_string.replace(char, '')

            output_file.write(output_smiles_string)
            smiles_list.append(output_smiles_string)

    print('Successfully converted FASTA into SMILES\n')

    return smiles_list


def get_smiles_list(smiles_path):
    """
    get_smiles_list creates and returns list of smiles strings. This function
    is used when the input file has smiles extension and not FASTA extension.
    Args:
        smiles_path (os.path): The path to the input smiles file that contains
        one or more sequences in smiles format.
    Returns:
        list: This list contains smiles strings as elements.
    """

    smiles_list = []
    with open(smiles_path, 'r') as input_file:
        smiles_list = input_file.readlines()

    for i in range(len(smiles_list)):
        for char in ['[', ']', '.']:
            smiles_list[i] = smiles_list[i].replace(char, '')

    return smiles_list


def plot_molecule_graph(G, labels, folder_name='graph', graph_num=None):
    """
    plot_molecule_graph creates a visual representation of the graph and saves
    it.
    Args:
        G (networkx.Graph): The graph representation of the molecule.
        labels (list): This list contains string labels that describe elements
        of the graph.
        folder_name (str, optional): The name of the directory where the image
        will be saved. Defaults to 'graph'.
        graph_num (int, optional): The optional integer that represents the
        number of the graph. It is used when the image has to be created for
        multiple molecules (graphs). Defaults to None.
    Returns:
        None: None
    """

    dirname = os.path.join(".", folder_name)
    create_dir(dirname)

    filename = os.path.join(dirname, str(graph_num) + '_graph.png')

    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=400)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    plt.savefig(filename)
    plt.close()

    return None


def direct_neighbors(vertices_list, graph):
    """
    direct_neighbors provides the direct neighbours for a collection of vertices
    without repeats.
    Args:
        vertices_list (list of int): Numbers representing vertices in graph.
        graph (networkx.graph): Graph object displaying the intermediary molecular
        graph.
    Returns:
        nb (list): Vertices which neighbor the vertices provided in the input in
        the input graph.
    """
    if type(vertices_list) == int:
        vertices_list = [vertices_list]

    nb = []
    for v in vertices_list:
        nb.extend([w for w in graph.neighbors(v)])
    nb = [*set(nb)]
    return nb


def dict_neighbors(vertex, graph, level=1, remove_carbon_neighbors=True):
    """
    dict_neighbors provides a dictionary, containing the neighboring vertices as
    a list for each level.
    Args:
        vertex (int): Vertex for which neighbors shall be calculated.
        graph (network.graph): The molecular graph.
        level (int, optional): Up to which level the neighborhoods shall be calculated.
        remove_carbon_neighbors (bool, optional): If set to True, any path that goes
        through another C in the carbon chain gets truncated. That way, no atom
        is included in the neighborhoods of multiple carbons.
    Returns:
        nb_dict (dict): Contains for each level a list of the neighboring vertices.
    """
    elements = nx.get_node_attributes(graph, 'element')
    carbon_list = [v for v in graph.nodes() if elements[v] == 'C']

    nb_dict = {0: [vertex]}

    for l in range(1, level + 1):
        curr_nb = nb_dict[l - 1]

        # remove other carbons from the neighbourhood search (but not the starting carbon)
        if remove_carbon_neighbors == True and l != 1:
            curr_nb = list(set(curr_nb) - set(carbon_list))

        # allows for quicker stop if no new neighbours
        if not curr_nb:
            for k in range(l, level + 1):
                nb_dict[k] = []
            return nb_dict

        next_nb = direct_neighbors(curr_nb, graph)

        if l >= 2:
            next_nb = list(set(next_nb).difference(set(nb_dict[l - 2])))

        nb_dict[l] = next_nb

    return nb_dict


def dict_neighbors_elements(nb_dict, elements):
    """
    dict_neighbours_elements turns the vertices in the dictionary of neighboorhoods
    into the corresponding elements.
    Args:
        nb_dict (dict): Contains for each level a list of the neighboring vertices.
        elements (list): List describing which vertex in the graph corresponds to
        which atom type.
    Returns:
        element_dict (dict): Contins for each level a list of the atoms contained in
        the neighborhood.
    """
    element_dict = {}
    for l in nb_dict.keys():
        element_dict[l] = [elements[v] for v in nb_dict[l]]

    return element_dict


def create_enc_df(level, graph, remove_carbon_neighbors=True, element_alphabet=['C', 'N', 'O', 'S']):
    """
    create_enc_df creates the atom count table.
    Args:
        level (int): Up to which level neighborhoods shall be collected.
        graph (networkx.graph): The molecular graph.
        remove_carbon_neighbors (bool, optional): If set to True, any path that goes
        through another C in the carbon chain gets truncated. That way, no atom
        is included in the neighborhoods of multiple carbons.
        element_alphabet (list of str, optional): Contains which elements are included in the 
        atom count table.
    Returns:
        enc_df (pandas.DataFrame): The atom count table.
    """
    elements = graph.nodes('element')

    # create dataframe of correct size
    carbon_nodes = [v for v in range(graph.number_of_nodes()) if graph.nodes[v]['element'] == 'C']
    nr_carbon = len(carbon_nodes)

    enc_dict = {}

    # populate dataframe column-wise

    for c_idx in range(nr_carbon):
        count_list = []

        c = carbon_nodes[c_idx]
        nb_dict = dict_neighbors(c, graph, level=level, remove_carbon_neighbors=remove_carbon_neighbors)
        element_dict = dict_neighbors_elements(nb_dict, elements)

        for l in range(1, level + 1):
            current_elements = element_dict[l]

            for e_idx in range(len(element_alphabet)):
                e = element_alphabet[e_idx]
                element_count = current_elements.count(e)
                count_list.append(element_count)

        enc_dict['C' + str(c_idx)] = count_list

    enc_df = pd.DataFrame.from_dict(enc_dict, orient='columns')
    enc_df.index = ['L' + str(l) + ' - ' + el for l in np.arange(1, level + 1)
                     for el in element_alphabet]

    return enc_df


def shift_padding(enc_df, col_nr, element_alphabet=['C', 'N', 'O', 'S']):
    """
    shift_padding adds columns of zeros to the encoding so that all encodings in
    one dataset have the same size.
    Args:
        enc_df (pandas.DataFrame): The original atom count table.
        col_nr (int): The desired number of columns.
        element_alphabet (list of str): Contains which elements are included in the 
        atom count table.
    Returns:
        enc_df_padd (pandas.DataFrame): The padded atom count table.
    """
    col_curr = len(enc_df.columns)
    col_diff = col_nr - col_curr

    row_nr = len(enc_df.index)

    level = row_nr // len(element_alphabet)
    row_names = ['L' + str(l) + ' - ' + el for l in np.arange(1, level + 1)
                     for el in element_alphabet]
    col_names = ['C' + str(i) for i in range(col_curr, col_nr)]
    zero_df = pd.DataFrame(np.zeros(shape=(row_nr, col_diff), dtype=int), index=row_names, columns=col_names)

    enc_df_padd = enc_df.join(zero_df)

    return enc_df_padd


def normalize_df(enc_df):
    """
    normalize_df normalized the dataframe so that its biggest entry is 1 and its
    smallest is 0.
    Args:
        enc_df (pandas.DataFrame): The original atom count table.
    Returns:
        enc_df / max_entry (pandas.DataFrame): The normalised atom count table.
    """
    max_entry = enc_df.to_numpy().max()
    return enc_df / max_entry


def create_dir(dirname):
    # helper to create directory if not yet existent
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except Exception as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise


def generate_imgs_from_encoding(normalized_encoding, print_progress=False, foldername=None, level=2,
                                alphabet_mode='without_hydrogen', element_alphabet=['C', 'N', 'O', 'S']):
    """
    generate_imgs_from_encoding generates images for all encodings.

    Args:
        normalized_encoding (dict): This dictionary contains the normalized
        encodings for each input file.
        binary_encoding (bool, optional): If this flag is True, the binary
        encoding is calculated. If it is False, discretized encoding is
        calculated. Defaults to True.
        folder_name (str, optional): This variable contains the name of the
        directory for encoding images. This directory is not created if
        generate_images is False. Defaults to "encoding_images".
        print_progress (bool, optional): If True, the progress of the
        calculation will be shown to the user. Defaults to False.

    Returns:
        None: None
    """
    if alphabet_mode == 'with_hydrogen':
        element_alphabet = ['H', 'C', 'N', 'O', 'S']

    if print_progress:
        clear_output(wait=True)
        progress = 0
        number_of_items = len(normalized_encoding)

    for name, encoding in normalized_encoding.items():

        if print_progress:
            clear_output(wait=True)
            progress += 1
            print('generating image {} of {}'.format(
                progress, number_of_items))

        x_fig_dim = max(encoding.shape[0] / 2, 6)
        y_fig_dim = max(encoding.shape[1] / 5, 10)
        plt.figure(figsize=(x_fig_dim, y_fig_dim))
        plt.title('Compound ' + str(name) + ', Encoding til level ' + str(level) + ' | element alphabet: ' + alphabet_mode, fontsize=18)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_color('black')
            ax.spines[axis].set_linewidth(3)

        dir_name = os.path.join('.', foldername)
        create_dir(dir_name)

        plt.imshow(np.array(encoding, dtype=float), cmap='gray_r')
        plt.xlabel('Carbon chain')
        plt.xticks(ticks=range(len(encoding.columns)), labels=encoding.columns)
        plt.ylabel('Atoms')
        plt.yticks(ticks=range(len(encoding.index)), labels=encoding.index)

        for l in range(1, level):
            plt.axline((0, l * len(element_alphabet) - 0.5), slope=0., linewidth=2, linestyle='dashed', color='black')

        subdir_name = os.path.join(dir_name, 'CENACT_level_' + str(level) + '_' + alphabet_mode)
        create_dir(subdir_name)
        filename = os.path.join(subdir_name, str(name) + '.jpg')

        plt.savefig(filename,bbox_inches='tight')
        plt.close()

    print('Saved images to folder ' + foldername, '\n')

    return None


def csv_export_cenact(normalized_encoding, output_path='encoding.csv'):
    """
    csv_export function exports the normalized encodings in a csv file.

    Args:
        normalized_encoding (dict): This dictionary contains the normalized
        encodings for each input file.
        classes (pd.DataFrame, optional): This DataFrame contains one column
        that holds the prediction class for each sequence. Defaults to
        pd.DataFrame.
        output_path (str, optional): This string is the name of the resulting
        csv file. Defaults to "encoding.csv".

    Returns:
        None: None
    """

    flatten_dict = {}
    for k in normalized_encoding.keys():
        flatten_dict[k] = normalized_encoding[k].to_numpy().flatten(order='F')
    encoding_as_df = pd.DataFrame.from_dict(flatten_dict, orient='index')
    encoding_as_df = encoding_as_df.reset_index(drop=True)

    encoding_as_df.to_csv(output_path, index=False)

    print('Successfully exported encodings to ', output_path, '\n')

    return None


def get_data_driven_element_alphabet(smiles_list):
    pattern = '|'.join(sorted(ELEMENTS_SET, key=len, reverse=True))
    matches = re.findall(pattern, ''.join(smiles_list))
    element_alphabet_set = set(matches)

    element_alphabet_set = element_alphabet_set - {'\t', '\n', '\r'}
    element_alphabet_unsorted = list(element_alphabet_set)
    element_alphabet = sorted(element_alphabet_unsorted, key=ELEMENTS_DICT.get)
    return element_alphabet


def cenact_encode(smiles_list, level, generate_imgs=False, alphabet_mode='without_hydrogen', print_progress=False,
                  output_path=None, foldername_encoding_vis='./CENACT_Encodings_Visualisation'):
    if alphabet_mode == 'with_hydrogen':
        element_alphabet = ['H', 'C', 'N', 'O', 'S']
    elif alphabet_mode == 'data_driven':
        element_alphabet = get_data_driven_element_alphabet(smiles_list)
    else:
        element_alphabet = ['C', 'N', 'O', 'S']

    enc_dict = {}
    padd_dict = {}

    max_carbon = 1

    for i, smiles in enumerate(smiles_list):
        mol = read_smiles(smiles, explicit_hydrogen=True)
        enc_df = create_enc_df(level=level, graph=mol, remove_carbon_neighbors=True, element_alphabet=element_alphabet)
        enc_dict[i] = enc_df
        len_chain = len(enc_df.columns)

        if len_chain > max_carbon:
            max_carbon = len_chain

    for i in enc_dict.keys():
        padd_dict[i] = shift_padding(enc_dict[i], max_carbon, element_alphabet=element_alphabet)

    if generate_imgs == True:
        # only create image for unpadded version of dataframe
        generate_imgs_from_encoding(enc_dict, print_progress=print_progress, foldername=foldername_encoding_vis,
                                    level=level, alphabet_mode=alphabet_mode, element_alphabet=element_alphabet)

    csv_export_cenact(padd_dict, output_path=output_path)
    return None


def main():
    program_name = 'CENACT'
    program_description = '''CENACT - Carbon-based Encoding of Neighbourhoods with Atom Count Tables'''

    input_help = 'A required path-like argument'

    alphabet_mode_help = '''An optional string argument that specifies which alphabet of elements
                         the algorithm should use: Possible options are only using the most abundant elements
                         in proteins and excluding hydrogen, i.e. C, N, O, S ('without_hydrogen'); using the
                         most abundant elements including hydrogen, i.e. H, C, N, O, S ('with_hydrogen'); and
                         using all elements which appear in the smiles strings of the dataset ('data_driven').
                         '''

    level_help = '''An optional integer argument that specifies the upper boundary of levels that should
                 be considered. Default: 2 (levels 1 and 2). Any integer returns neighbourhoods up to
                 that level.
                 '''

    image_help = '''An optional integer argument that specifies whether images should be created or not.
                 Default: 0 (without images).
                 '''

    graph_help = '''An optional integer argument that specifies whether a graph representation should be
                 created or not. Default: 0 (without representation). The user should provide the number
                 between 1 and the number of sequences in the parsed input file. Example: if number 5 is
                 parsed for this option, a graph representation of the 5th sequence of the input file
                 shall be created and placed in the corresponding images folder.
                 '''

    output_dir_name = "CENACT_Encodings"

    output_path = os.path.join('.', output_dir_name)
    output_help = '''An optional path-like argument. For parsed paths, the directory must exist beforehand.
                     Default: ''' + output_path

    input_error = 'Input file path is bad or the file does not exist'
    input_extension_error = '''The input file should be FASTA or SMILES.
                            Allowed extensions for FASTA: .fa, .fasta.
                            Allowed extensions for SMILES: .smi, .smiles.
                            The tool also supports any uppercase combination
                            of the aforementioned extensions.
                            '''
    graph_error = '''Graph should be an integer >=1 and <=number of sequences in the input file'''
    output_error = '''Output directory path is bad or the directory does not exist'''

    argument_parser = argparse.ArgumentParser(
        prog=program_name, description=program_description)

    # Adding arguments
    allowed_alphabet_mode = ['without_hydrogen', 'with_hydrogen', 'data_driven']
    allowed_images = [0, 1]

    argument_parser.add_argument('input_file', type=pathlib.Path, help=input_help)
    argument_parser.add_argument('--alphabet_mode', type=str, help=alphabet_mode_help,
                                 choices=allowed_alphabet_mode, default='without_hydrogen')
    argument_parser.add_argument('--level', type=int, help=level_help, default=2)
    argument_parser.add_argument('--image', type=int, help=image_help, choices=allowed_images, default=0)
    argument_parser.add_argument('--show_graph', type=int, help=graph_help)
    argument_parser.add_argument('--output_path', type=pathlib.Path, help=output_help)

    # Parsing arguments
    arguments = argument_parser.parse_args()

    # Additional argument inspection
    if not os.path.exists(arguments.input_file):
        argument_parser.error(input_error)

    if arguments.show_graph is not None:
        if arguments.show_graph <= 0:
            argument_parser.error(graph_error)

    if arguments.output_path is not None:
        # Output path is the user-settable path
        output_dir = os.path.join('.', str(arguments.output_path))
    else:
        # Output path is the default path
        output_dir = os.path.join('.', output_dir_name)

    # Create the results directory
    create_dir(output_dir)

    # STEP 1: Open the input file and check the format
    input_file_dir, input_file = os.path.split(arguments.input_file)
    input_file_name, input_file_extension = os.path.splitext(
        input_file)

    input_file_extension = input_file_extension.strip().lower()

    if input_file_extension not in ['.smi', '.smiles', '.fa', '.faa', '.fasta']:
        argument_parser.error(input_extension_error)

    # STEP 2: Define important variables. Also get the number of sequences in
    # a file. Do conversion to SMILES format if FASTA is provided as an input
    print('\n============================================================')
    print('                           CENACT                           ')
    print('============================================================')

    generate_imgs = True if arguments.image == 1 else False
    level = arguments.level
    alphabet_mode = arguments.alphabet_mode

    output_path = os.path.join(output_dir, 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')

    if input_file_extension in ['.smi', '.smiles']:
        input_smiles_path = arguments.input_file
        smiles_list = get_smiles_list(input_smiles_path)
    elif input_file_extension in ['.fa', '.faa', '.fasta']:
        smiles_path = os.path.join(input_file_dir, 'smiles.smi')
        smiles_list = convert_fasta_to_smiles(
            arguments.input_file, smiles_path)

    num_of_lines = len(smiles_list)

    # STEP 3: One more check if show_graph is set to 1
    # This step checks if the user-inputted number is lower than the number
    # of lines in the SMILES or FASTA file
    if arguments.show_graph is not None:
        if arguments.show_graph > num_of_lines:
            argument_parser.error(graph_error)

    # STEP 4: Encode and export molecules
    # Possibly generate images and the graph, if selected
    cenact_encode(smiles_list, level=level, generate_imgs=generate_imgs, alphabet_mode=alphabet_mode,
                  print_progress=False, output_path=output_path,
                  foldername_encoding_vis='CENACT_Encodings_Visualisation')

    print('============================================================\n')

    return None


if __name__ == '__main__':
    main()
