import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def rekomendasi_hotel(data): 
        output_dataset = pd.read_csv("https://storage.googleapis.com/data-hotel/list-hotel/output_dataset.csv", encoding='unicode_escape')
        df = pd.read_csv("https://storage.googleapis.com/data-hotel/list-hotel/data_preprocessing.csv", encoding='unicode_escape')

        # Splitting Section
        X = df[['stars','reviews','harga', 'Shuttle Service', 'Sports and Recreations', 'Kids and Pets', 'Transportation', 'Connectivity', 'Accessibilty', 'Things to Do', 'General', 'Public Facilities', 'Nearby Facilities', 'Business Facilities', 'In-room Facilities', 'Hotel Services', 'Food and Drinks', 'Fast Food', 'Shop & Gifts', 'Business', 'Transportation Hub', 'Casual Dining', 'Nightlife', 'Park & Zoo', 'Public Service', 'Arts & Sciences', 'Fine Dining', 'Sport', 'Quick Bites', 'Education', 'Street Food', 'Activity & Games', 'Cafe', 'Entertainment', 'Food Court', 'Sight & Landmark' ]]
        y = df['rating']


        dev_X, val_X, dev_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
        train_dataset = df.loc[:70]
        test_dataset = df.drop(train_dataset.index)
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('rating')
        test_labels = test_features.pop('rating')
        # End of Splitting Section

        # Modeling
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        regularizer = 0.000001
        dropout = 0.25
        schedul = -0.0001
        lr = 0.001

        optimizer = tf.optimizers.Adam(learning_rate=lr)
            
        def scale_model(norm):
              model = keras.Sequential([
              norm,
              layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
              layers.Dropout(dropout),
              layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
              layers.Dropout(dropout),
              layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
              layers.Dropout(dropout),
              layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
              layers.Dropout(dropout),
              layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),
              layers.Dropout(dropout),
              layers.Dense(1)
              ])    
              return model
        
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(schedul)

        def DNN_Pipeline (model):
            
            model.compile(optimizer= optimizer, loss='mean_absolute_error')
            
            history = model.fit(
                      train_features,
                      train_labels,
                      validation_split=0.2,
                      callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler),
                      verbose=0, epochs=300)
            
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 3])
            plt.xlabel('Epoch')
            plt.ylabel('Error [MPG]')
            plt.legend()
            plt.grid(True)
            print(hist.tail(10))
            
            return model

        scale = DNN_Pipeline(scale_model(normalizer))

        scale.predict(train_features)

        df_rank = train_dataset
        df_rank['predict_score'] = scale.predict(train_features)

        new_sorted = df_rank.sort_values(by=['predict_score'], ascending=False)

        print(new_sorted)

        # Input Parameter
        test = pd.read_json(json.dumps(data))
        test = test['interest']
        weights = []
        for index, row in new_sorted.iterrows():
            for i in range(len(test)):
                weights.append(new_sorted['{}'.format(test.iat[i])])
            break
        for i in range(len(test)):
            # weights[i] = weights[i] / new_sorted['{}'.format(test.iat[i])].max()
            weights[i] = weights[i] * 2
        
        df_test = df[test]
        df_test['predict_score'] = new_sorted['predict_score']
        df_test = df_test[:70]
        df_test['final_score'] = df_test.sum(axis=1)

        output_dataset['final_score'] = df_test['final_score']
        output_dataset = output_dataset.sort_values(by=['final_score'], ascending=False)

        # return data dalam bentuk json
        return output_dataset.to_json(orient ='table')
