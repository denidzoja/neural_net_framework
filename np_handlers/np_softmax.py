
import numpy as np

from .base import NumpyHandlerBase


def get_softmax_layer_np_handler_cls():
    return NpSoftmaxLayerHandler

class NpSoftmaxLayerHandler(NumpyHandlerBase):
    def __init__(self, parent_obj):
        self.parent_obj = parent_obj

    def get_softmax_probabilities_from_inputs(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        return probabilities
    
    def convert_one_hot_to_discrete_labels(self, true_values):
        if len(true_values.shape) == 2:
            true_values = np.argmax(true_values, axis=1)
        return true_values