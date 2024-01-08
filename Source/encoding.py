import os
import pathlib
import cenact
from joblib import Parallel, delayed

def encode_all_datasets():
    datasets_path = os.path.join('..', 'Data', 'Original_datasets')
    datasets_folder = pathlib.Path(datasets_path)
    datasets_list = list(datasets_folder.iterdir())
    datasets_list = [str(path).split("\\")[-1] for path in datasets_list]

    encodings_path = os.path.join('..', 'Data', 'Encodings')
    if os.path.exists(encodings_path) == False:
        os.mkdir(encodings_path)

    for dataset in datasets_list:
        fasta_path = os.path.join(dataset, 'seqs.fasta')
        smiles_path = os.path.join(dataset, 'seqs.smiles')
        dataset_filename = os.path.basename(dataset)
        encoding_data_path = os.path.join(encodings_path, dataset_filename)
        if os.path.exists(encoding_data_path) == False:
            os.mkdir(encoding_data_path)

        smiles_list = cenact.convert_fasta_to_smiles(fasta_path, smiles_path)

        for level in [1,2]:
            for alphabet_mode in ['without_hydrogen', 'with_hydrogen', 'data_driven']:
                output_path = os.path.join(encoding_data_path, 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')
                cenact.cenact_encode(smiles_list, level=level, generate_imgs=False, alphabet_mode=alphabet_mode,
                                    print_progress=False, output_path=output_path,
                                    foldername_encoding_vis='Novel_Encodings_Visualisation')
                
# version of function that only calculates encoding if there is no folder of the name yet

def encode_all_datasets_var():
    datasets_path = os.path.join('..', 'Data', 'Original_datasets')
    datasets_folder = pathlib.Path(datasets_path)
    datasets_list = list(datasets_folder.iterdir())
    datasets_list = [str(path).split("\\")[-1] for path in datasets_list]


    encodings_path = os.path.join('..', 'Data', 'Encodings')
    if os.path.exists(encodings_path) == False:
        os.mkdir(encodings_path)

    for dataset in datasets_list:
        fasta_path = os.path.join(dataset, 'seqs.fasta')
        smiles_path = os.path.join(dataset, 'seqs.smiles')
        dataset_filename = os.path.basename(dataset)
        encoding_data_path = os.path.join(encodings_path, dataset_filename)

        if os.path.exists(encoding_data_path) == False:
            os.mkdir(encoding_data_path)
            if os.path.exists(smiles_path) == False:
                smiles_list = cenact.convert_fasta_to_smiles(fasta_path, smiles_path)
            else:
                smiles_list = cenact.get_smiles_list(smiles_path)

            for level in [1,2]:
                for alphabet_mode in ['without_hydrogen', 'with_hydrogen', 'data_driven']:
                    output_path = os.path.join(encoding_data_path, 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')
                    cenact.cenact_encode(smiles_list, level=level, generate_imgs=False, alphabet_mode=alphabet_mode,
                                        print_progress=False, output_path=output_path,
                                        foldername_encoding_vis='Novel_Encodings_Visualisation')

###################################################
######## PARALLELISED VERSION (NOT TESTED) ########
###################################################
def encode_parallel(smiles, level, element_alphabet):
    mol = cenact.read_smiles(smiles, explicit_hydrogen=True)
    enc_df = cenact.create_enc_df(level=level, graph=mol, remove_carbon_neighbors=True, element_alphabet=element_alphabet)
    return enc_df

def encode_all_datasets_parallel():
    datasets_path = os.path.join('..', 'Data', 'Original_datasets')
    datasets_folder = pathlib.Path(datasets_path)
    datasets_list = list(datasets_folder.iterdir())
    datasets_list = [str(path).split('\\')[-1] for path in datasets_list]

    encodings_path = os.path.join('..', 'Data', 'Encodings')
    if os.path.exists(encodings_path) == False:
        os.mkdir(encodings_path)

    for dataset in datasets_list:
        fasta_path = os.path.join(dataset, 'seqs.fasta')
        smiles_path = os.path.join(dataset, 'seqs.smiles')
        dataset_filename = os.path.basename(dataset)
        encoding_data_path = os.path.join(encodings_path, dataset_filename)
        if os.path.exists(encoding_data_path) == False:
            os.mkdir(encoding_data_path)

        if os.path.exists(smiles_path) == False:
            smiles_list = cenact.convert_fasta_to_smiles(fasta_path, smiles_path)
        else:
            smiles_list = cenact.get_smiles_list(smiles_path)

        for level in [1,2]:
            for alphabet_mode in ['without_hydrogen', 'with_hydrogen', 'data_driven']:
                output_path = os.path.join(encoding_data_path, 'novel1_level_' + str(level) + '_' + alphabet_mode + '.csv')

                if alphabet_mode == 'with_hydrogen':
                    element_alphabet = ['H', 'C', 'N', 'O', 'S']
                elif alphabet_mode == 'data_driven':
                    element_alphabet = cenact.get_data_driven_element_alphabet(smiles_list)
                else:
                    element_alphabet = ['C', 'N', 'O', 'S']

                padd_dict = {}

                enc_list = Parallel(n_jobs=-1)(delayed(encode_parallel)(smiles, level=level, element_alphabet=element_alphabet) for smiles in smiles_list)
                max_carbon = max([len(enc_df.columns) for enc_df in enc_list])

                for i in range(enc_list):
                    padd_dict[i] = cenact.shift_padding(enc_list[i], max_carbon, element_alphabet=element_alphabet)

                cenact.csv_export_cenact(padd_dict, output_path=output_path)

if __name__ == '__main__':
    encode_all_datasets_var()