import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
from ela import ela
from prnu import prnu

with open('ela_history_dict.pickle', 'rb') as handle:
        history = pickle.load(handle)

def plot(train_metrics, val_metrics, title, y_label, save=True):
    plt.plot(train_metrics)
    plt.plot(val_metrics)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig(f'../poster_pics/ela_{y_label}.png')
    plt.show()

def plot_loss():
    plot(history['loss'], history['val_loss'], 'ela model loss', 'loss')

def plot_acc():
    plot(history['accuracy'], history['val_accuracy'], 'ela model accuracy', 'accuracy')

def print_acc():
    print('train acc: ' + str(history['accuracy'][-1]))
    print('val acc: ' + str(history['val_accuracy'][-1]))

def print_precision():
    print('train precision: ' + str(history['precision'][-1]))
    print('val precision: ' + str(history['val_precision'][-1]))

def print_recall():
    print('train recall: ' + str(history['recall'][-1]))
    print('val recall: ' + str(history['val_recall'][-1]))

def print_f1():
    print('train f1: ' + str(history['f1_score'][-1]))
    print('val f1: ' + str(history['val_f1_score'][-1]))

plot_loss()
# plot_acc()
print_acc()
print_precision()
print_recall()
# print_f1()

def show_ela(image_path):
     ela_im = ela(image_path)
     ela_im = cv2.cvtColor(ela_im, cv2.COLOR_BGR2GRAY)
     ela_im = cv2.equalizeHist(ela_im)
     plt.gray()
     plt.imsave('../poster_pics/ela_fake_img2.png', ela_im)
     plt.imshow(ela_im)
     plt.show()

# show_ela('../data/train/REAL/0013 (4).jpg')
# show_ela('../data/train/REAL/0001.jpg')
# show_ela('../data/train/FAKE/1048 (3).jpg')