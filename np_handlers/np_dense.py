
import numpy as np

from .base import NumpyHandlerBase


def get_dense_layer_np_handler_cls():
    return NpDenseLayerHandler

class NpDenseLayerHandler(NumpyHandlerBase):

    def __init__(self, parent_obj):
        self.parent_obj = parent_obj
    
    def forward_pass(self, inputs, weights, biases):
        return np.dot(inputs, weights) + biases

    def calculate_dweights(self, inputs, input_grad):
        return np.dot(inputs.T, input_grad)
    
    def calculate_dinputs(self, input_grad, weights):
        return np.dot(input_grad, weights.T)
    
    def calculate_dbiases(self, input_grad):
        return np.sum(input_grad, axis=0, keepdims=True)
    
    def initialize_weights(self, num_inputs, num_neurons):
        return 0.01 * np.random.randn(num_inputs, num_neurons)
    
    def initialize_biases(self, num_neurons):
        return np.zeros((1, num_neurons))