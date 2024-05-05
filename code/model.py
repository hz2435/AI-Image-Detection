import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from preprocess import get_data
import os
import pickle

class Model:
    def __init__(self):
        if (os.getcwd().split('/')[-1] != 'code'):
            raise IsADirectoryError("Must be in code directory")
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 1)), #change shape to (32, 32, 1) for ELA to work!!!!!! 
                                                      #change shape to (32, 32, 3) for PRNU to work!!!!!!
            
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0, 25),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0, 25),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0, 25),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0, 25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0, 5),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.train_and_eval()

    def train_and_eval(self):
        train_data, train_labels, test_data, test_labels = get_data()

        num_samples = len(train_data)
        indices = tf.range(num_samples)
        shuffled_indices = tf.random.shuffle(indices)

        train_data = tf.gather(train_data, shuffled_indices)
        train_labels = tf.gather(train_labels, shuffled_indices)

        num_samples = len(test_data)
        indices = tf.range(num_samples)
        shuffled_indices = tf.random.shuffle(indices)

        test_data = tf.gather(test_data, shuffled_indices)
        test_labels = tf.gather(test_labels, shuffled_indices)

        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=2, average="micro")])

        hist = self.model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
        print(hist.history.keys())

        with open('history_dict.pickle', 'wb') as handle:
            pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        metric_names = self.model.metrics_names
        test_metrics = self.model.evaluate(test_data, test_labels)
        for i in len(metric_names):
            print(metric_names[i] + ": " + test_metrics[i])
        # print("Test Accuracy:", test_accuracy)
        # print("Test Loss:", test_loss)

        self.model.save("saved_model_metrics.h5")


Model()