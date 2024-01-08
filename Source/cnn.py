import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
import keras.backend as K
import tensorflow as tf
import os
import pathlib

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_positives = tf.cast(true_positives, dtype=tf.float64)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = tf.cast(possible_positives, dtype=tf.float64)
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_positives = tf.cast(true_positives, dtype=tf.float64)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_positives = tf.cast(predicted_positives, dtype=tf.float64)
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

datasets_path = os.path.join('..', 'Data', 'Encodings')
datasets_folder = pathlib.Path(datasets_path)
datasets_list = list(datasets_folder.iterdir())
datasets_list = [os.path.basename(dataset) for dataset in datasets_list]

f1_score_df = pd.DataFrame(np.nan, index=datasets_list, columns=range(5))

epochs = 100

pre_path = os.path.join('..', 'Results')
if os.path.exists(pre_path) == False:
    os.mkdir(pre_path)

results_path = os.path.join('..', 'Results', 'CNN')
if os.path.exists(results_path) == False:
    os.mkdir(results_path)

f1_score_path = os.path.join(results_path, 'f1_score_level_' + str(2) + '_' + 'with_hydrogen' + '.csv')

for level in [2]:
    for alphabet_mode in ['with_hydrogen']:
        for data_idx in range(len(datasets_list)):
            dataset = datasets_list[data_idx]
            print("Running dataset", data_idx + 1, "/", len(datasets_list), flush=True)

            enc_path = os.path.join('..', 'Data', 'Encodings', dataset, 'CENACT_level_' + str(level) + '_' + alphabet_mode + '.csv')
            classes_path = os.path.join('..', 'Data', 'Original_datasets', dataset, 'classes.txt')

            X = pd.read_csv(enc_path)
            y = pd.read_csv(classes_path, header=None)
            y = y.astype("float")
            y = y.to_numpy().ravel()

            print("Dataset loaded.", flush=True)

            # Normalize the data
            X = X / 2

            # Reshape the data to image
            dim = (10, len(X.columns) // 10) #need to adapt number depending on the mode
            X = X.values.reshape(-1,dim[0],dim[1],1, order = "F") #reshape shapes back to image correctly (ie first 8 entries become first column)
            print("Dataset normalised and reshaped.", flush=True)

            for i in range(5):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=i, shuffle=True, stratify=None)
                X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, train_size=0.5, random_state=i, shuffle=True, stratify=None)

                batch_size = 32

                # Build model
                model = Sequential()
                #
                model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
                                activation ='relu', input_shape = (dim[0],dim[1],1)))
                model.add(MaxPool2D(pool_size=(2,2)))
                model.add(Dropout(0.25))
                #
                model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
                                activation ='relu'))
                model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
                model.add(Dropout(0.25))
                #
                # fully connected
                model.add(Flatten())
                model.add(Dense(256, activation = "relu"))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation="sigmoid"))

                # Define the optimizer
                optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

                # Compile the model
                model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"]) # AS OF YET DOES NOT OPTIMISE FOR F1-SCORE!

                # Define and train data generator
                datagen = ImageDataGenerator()
                datagen.fit(X_train)

                # Fit the model
                history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs = epochs,
                                              validation_data = (X_val, y_val), steps_per_epoch=X_train.shape[0] // batch_size,
                                              verbose=0)

                y_pred_soft = model.predict(X_test)
                y_pred_hard = tf.cast(K.greater_equal(y_pred_soft, 0.5), dtype = tf.int64)
                y_pred_hard = tf.reshape(y_pred_hard, shape=y_pred_hard.shape[0])
                y_test = tf.cast(y_test, dtype = tf.int64)
                f1_score = f1(y_test, y_pred_hard).numpy()
                print("test f1-score:", f1_score, flush=True)

                f1_score_df.iat[data_idx, i] = f1_score
                f1_score_df.to_csv(f1_score_path, index=True, header=True)


