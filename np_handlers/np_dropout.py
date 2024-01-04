
import numpy as np

from .base import NumpyHandlerBase


def get_dropout_layer_np_handler_cls():
    return NpDropoutLayerHandler

class NpDropoutLayerHandler(NumpyHandlerBase):
    def __init__(self, parent_obj):
        self.parent_obj = parent_obj
    
    def calculate_binary_mask(self, inputs):
        return np.random.binomial(1, self.parent_obj.retention_rate,
                                  size=inputs.shape) / self.parent_obj.retention_rate