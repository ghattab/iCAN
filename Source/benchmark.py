import os
import pandas as pd
import pathlib
import cenact
from timeit import default_timer as timer
import numpy as np

runs = 5
benchmark_path = os.path.join('..', 'Results', 'Benchmark', 'benchmark_runs_' + str(runs) + '.csv')

def benchmark():
    datasets_path = os.path.join('..', 'Data', 'Original_datasets')
    datasets_folder = pathlib.Path(datasets_path)
    datasets_list = list(datasets_folder.iterdir())

    encodings_path = os.path.join('..', 'Data', 'Encodings')
    if os.path.exists(encodings_path) == False:
        os.mkdir(encodings_path)

    benchmark_df = pd.DataFrame(index=[str(path).split("/")[-1] for path in datasets_list], columns=['level_1_without_hydrogen', 'level_1_with_hydrogen', 'level_1_data_driven',
                                                              'level_2_without_hydrogen', 'level_2_with_hydrogen', 'level_2_data_driven'])

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

                col_name = 'level_' + str(level) + '_' + alphabet_mode
                runtimes_list = []
                for run in range(runs):
                    output_path = os.path.join('.', 'DELETE', 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')

                    timer_start = timer()

                    cenact.cenact_encode(smiles_list, level=level, generate_imgs=False, alphabet_mode=alphabet_mode,
                                        print_progress=False, output_path=output_path,
                                        foldername_encoding_vis='Novel_Encodings_Visualisation')
                
                    timer_end = timer()

                    runtime = timer_end - timer_start
                    runtimes_list.append(runtime)
                    os.remove(os.path.join('.', 'DELETE', 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv'))
                
                median_runtime = np.median(runtimes_list)
                benchmark_df.loc[str(dataset).split("/")[-1], col_name] = median_runtime
        
    benchmark_df.to_csv(benchmark_path)

if os.path.exists(os.path.join('.', 'DELETE')) == False:
    os.mkdir(os.path.join('.', 'DELETE'))
benchmark()
os.rmdir(os.path.join('.', 'DELETE'))