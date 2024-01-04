
from __future__ import annotations

import typing as t
import logging

import pickle

from loss.loss_base import LossBase
from loss.categorical_cross_entropy import CategoricalCrossEntropyLoss

from layers.input import get_dflt_input_layer_cls
from layers.activation_funcs.softmax import SoftmaxActivation
from layers.activation_funcs.composite import SoftmaxActivationCategoricalCrossEntropyLoss


class NeuralNetModelBase:
    pass


class NeuralNetModel(NeuralNetModelBase):

    def __init__(self, layers=None, input_layer=None, model_type='rough'):
        if layers is None:
            self.layers = list()

        if input_layer is None:
            input_layer_cls = get_dflt_input_layer_cls()
            self.input_layer = input_layer_cls()

        self.trainable_layers = list()
        self.combined_output_classifier = None
        self.model_type = model_type

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_model_params(self, loss=None, optimizer=None, accuracy=None):
        if loss is None:
            self.loss = LossBase()
        else:
            self.loss = loss

        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def get_parameters(self):
        parameters = list()
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def finalize(self):
        self._initialize_layers()

        if self.loss is not None:
            self.loss.initialize_loss_func(self.trainable_layers)

        if self._is_valid_combined_classifier():
            self.combined_output_classifier = \
                SoftmaxActivationCategoricalCrossEntropyLoss()

    def train(self, input_values, true_values, epochs=1,
              print_every=1, validation_data=None):

        train_steps = 1

        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            self._reset_loss_and_accuracy()
            output = self.forward(input_values, training=True)

            data_loss, regularization_loss = \
                self.loss.calculate(output, true_values,
                                    include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.final_activation_layer.predictions(
                                output)
            accuracy = self.accuracy.calculate(predictions,
                                                true_values)

            self.backward(output, true_values)

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            print(f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.validate_data(*validation_data)
            
            if self.model_type == 'rough' and epoch_accuracy > 0.85:
                print("breaking! 0.85 accuracy reached. ")
                break

    def validate_data(self, X_val, y_val):

        validation_steps = 1
        self.loss.reset_state()
        self.accuracy.reset_state()

        for step in range(validation_steps):
            output = self.forward(X_val, training=False)
            self.loss.calculate(output, y_val)
            predictions = self.final_activation_layer.predictions(
                              output)
            self.accuracy.calculate(predictions, y_val)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def predict(self, input_values):
        output_list = list()
        output = self.forward(input_values, training=False)
        output_list.append(output)
        return output_list

    def forward(self, inputs, training):
        self.input_layer.forward_pass(inputs)

        for layer in self.layers:
            layer.forward_pass(layer.prev.output, training)

        return layer.output

    def backward(self, output, true_values):
        if self.combined_output_classifier is not None:
            self._perform_backward_pass_for_combined_case(output, true_values)
        else:
            self._perform_backward_pass(output, true_values)

    def _perform_backward_pass_for_combined_case(self, output, true_values):
        self.combined_output_classifier.backward_pass(output, true_values)
        self.layers[-1].dinputs = \
            self.combined_output_classifier.dinputs

        for layer in reversed(self.layers[:-1]):
            layer.backward_pass(layer.next.dinputs)
        return

    def _perform_backward_pass(self, output, true_values):
        self.loss.backward(output, true_values)
        for layer in reversed(self.layers):
            layer.backward_pass(layer.next.dinputs)

    def _initialize_layers(self):
        num_layers = len(self.layers)

        for i in range(num_layers):
            curr_layer = self.layers[i]
            if i == 0:
                curr_layer.prev = self.input_layer
                curr_layer.next = self.layers[i+1]

            elif i < num_layers - 1:
                curr_layer.prev = self.layers[i-1]
                curr_layer.next = self.layers[i+1]

            else:
                curr_layer.prev = self.layers[i-1]
                curr_layer.next = self.loss
                self.final_activation_layer = curr_layer

            if self._is_trainable_layer(curr_layer):
                self.trainable_layers.append(curr_layer)

    def _reset_loss_and_accuracy(self):
        self.loss.reset_state()
        self.accuracy.reset_state()

    def _is_valid_combined_classifier(self):
        return isinstance(self.layers[-1], SoftmaxActivation) and \
           isinstance(self.loss, CategoricalCrossEntropyLoss)

    def _is_trainable_layer(self, layer):
        return hasattr(layer, 'weights') and hasattr(layer, 'biases')