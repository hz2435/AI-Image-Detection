import tensorflow as tf  
import matplotlib.pyplot as plt
from preprocess import get_data
import os

class Model:
    def __init__(self):
        if (os.getcwd().split('/')[-1] != 'code'):
            raise IsADirectoryError("Must be in code directory")
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            
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
              metrics=['accuracy'])

        self.model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

        test_loss, test_accuracy = self.model.evaluate(test_data, test_labels)
        print("Test Accuracy:", test_accuracy)

        self.model.save("saved_model.h5")


Model()