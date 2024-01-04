

import sys

import numpy as np

import cv2

import os

from pathlib import Path

from model import NeuralNetModel
from loss.categorical_cross_entropy import CategoricalCrossEntropyLoss
from accuracy.categorical import CategoricalAccuracy
from optimizers.sgd import SGDOptimizer
from layers import (DenseLayer,
                    DropoutLayer)
from layers.activation_funcs.relu import Activation_ReLU
from layers.activation_funcs.softmax import SoftmaxActivation

def add_prj_root_dir_to_path():
    root_dir = Path(__file__).parent
    sys.path.append(root_dir)


def load_mnist_dataset(dataset, path):
    X = list()
    y = list()

    labels = os.listdir(os.path.join(path, dataset))
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def add_model_layers(model, input_data):
    model.add_layer(DenseLayer(input_data.shape[1], 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add_layer(Activation_ReLU())
    model.add_layer(DenseLayer(128, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add_layer(Activation_ReLU())
    model.add_layer(DenseLayer(128, 10))
    model.add_layer(SoftmaxActivation())

def set_model_params(model):
    loss_model = CategoricalCrossEntropyLoss()
    optimizer_model = SGDOptimizer(learning_rate=0.05, decay=1e-4, momentum=0.9)
    accuracy_model = CategoricalAccuracy()

    model.set_model_params(loss=loss_model, optimizer=optimizer_model, accuracy=accuracy_model)
    model.finalize()

def shuffle_training_data(input_data, classes):
    keys = np.array(range(input_data.shape[0]))
    np.random.shuffle(keys)
    input_data = input_data[keys]
    classes = classes[keys]
    return input_data, classes

def create_fashion_mnist_model(train_data, train_classes, valid_data, valid_classes):
    model = NeuralNetModel(model_type='accurate')

    add_model_layers(model, train_data)
    set_model_params(model)
    rough_params = Path(__file__).parent / 'fashion_mnist_rough'
    model.load_parameters(rough_params)
    model.train(train_data, train_classes, validation_data=(valid_data, valid_classes),
                epochs=300, print_every=100)
    
    output_file = Path(__file__).parent / 'fashion_mnist_accurate'
    model.save_parameters(output_file)

def flatten_images(input_data):
    input_data_flattened = (input_data.reshape(input_data.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    return input_data_flattened