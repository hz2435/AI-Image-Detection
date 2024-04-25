import numpy as np 
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0 
    return img

def calculate_prnu(images):
    prnu = np.mean(images, axis=0) 
    return prnu

def extract_prnu_residuals(images, prnu):
    prnu_residuals = [img - prnu for img in images]
    return np.array(prnu_residuals)

def get_images():
    fake_test = []
    for image in os.listdir("../data/test/FAKE"):
        fake_test.append(preprocess_image(os.path.join("../data/test/FAKE", image)))

    real_test = []
    for image in os.listdir("../data/test/REAL"):
        real_test.append(preprocess_image(os.path.join("../data/test/REAL", image)))

    fake_train = []
    for image in os.listdir("../data/train/FAKE"):
        fake_train.append(preprocess_image(os.path.join("../data/train/FAKE", image)))

    real_train = []
    for image in os.listdir("../data/train/REAL"):
        real_train.append(preprocess_image(os.path.join("../data/train/REAL", image)))

    return fake_test, real_test, fake_train, real_train

def get_data():
    fake_test, real_test, fake_train, real_train = get_images()

    train_data = fake_train + real_train
    train_labels = np.array([0 for _ in range(len(fake_train))] + [1 for _ in range(len(real_train))])

    test_data = fake_test + real_test
    test_labels = np.array([0 for _ in range(len(fake_test))] + [1 for _ in range(len(real_test))])

    prnu_train = calculate_prnu(train_data)
    train_data = np.array(extract_prnu_residuals(train_data, prnu_train))

    prnu_test = calculate_prnu(test_data)
    test_data = np.array(extract_prnu_residuals(test_data, prnu_test))

    return train_data, train_labels, test_data, test_labels