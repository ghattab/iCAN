from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
import pathlib

datasets_path = os.path.join('..', 'Data', 'Encodings')
datasets_folder = pathlib.Path(datasets_path)
datasets_list = list(datasets_folder.iterdir())
datasets_list = [os.path.basename(dataset) for dataset in datasets_list]

mean_accuracy_df = pd.DataFrame(np.nan, index=datasets_list, columns=range(50))
f1_score_df = pd.DataFrame(np.nan, index=datasets_list, columns=range(50))
proba_class1_df = pd.DataFrame(np.nan, index=datasets_list, columns=range(50))

def get_splits(dataset_size):
    return int(np.round(dataset_size / (0.2 * dataset_size)))

for level in [2]:
    for alphabet_mode in ['without_hydrogen', 'with_hydrogen', 'data_driven']:
        for data_idx in range(len(datasets_list)):
            dataset = datasets_list[data_idx]
            print('Running dataset', data_idx + 1, '/', len(datasets_list))

            encoding_data_path = os.path.join('..', 'Data', 'Encodings', dataset,
                                              'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')
            classes_path = os.path.join('..', 'Data', 'Original_datasets',
                                        dataset, 'classes.txt')

            X = pd.read_csv(encoding_data_path)
            y = pd.read_csv(classes_path, header=None)
            y = y.astype('category')
            y = y.to_numpy().ravel()

            cv = RepeatedStratifiedKFold(n_splits=get_splits(X.shape[0]), n_repeats=10, random_state=42)
            
            for i, (train_index, test_index) in enumerate(cv.split(X, y)):
                X_train, y_train = X.iloc[train_index,:], y[train_index]
                X_test, y_test = X.iloc[test_index,:], y[test_index]
                rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42)
                rfc.fit(X_train, y_train)
                mean_accuracy = rfc.score(X_test, y_test)
                y_pred = rfc.predict(X_test)
                f1 = f1_score(y_test, y_pred)

                mean_accuracy_df.iat[data_idx, i] = mean_accuracy
                f1_score_df.iat[data_idx, i] = f1

        results_path = os.path.join('..', 'Results', 'csv')
        if os.path.exists(results_path) == False:
            os.mkdir(results_path)

        mean_accuracy_path = os.path.join(results_path,
                                          'mean_accuracy_level_' + str(level) + '_' + alphabet_mode + '.csv')
        f1_score_path = os.path.join(results_path, 'f1_score_level_' + str(level) + '_' + alphabet_mode + '.csv')

        mean_accuracy_df.to_csv(mean_accuracy_path, index=True, header=True)
        f1_score_df.to_csv(f1_score_path, index=True, header=True)