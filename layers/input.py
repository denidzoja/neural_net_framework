
from .base import LayerBase


def get_dflt_input_layer_cls():
    return InputLayer

class InputLayer(LayerBase):

    def forward_pass(self, inputs):
        self.output = inputs

    def backward_pass(self):
        pass