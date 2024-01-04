
from np_handlers.np_dropout import get_dropout_layer_np_handler_cls

from .base import LayerBase


class DropoutLayer(LayerBase):

    def __init__(self, dropout_rate):
        self.np_handler = get_dropout_layer_np_handler_cls()(self)
        self.retention_rate = 1 - dropout_rate

    def forward_pass(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.nodes_to_dropout = self.np_handler.calculate_binary_mask(inputs)
        self.output = inputs * self.nodes_to_dropout

    def backward_pass(self, input_grad):
        self.dinputs = input_grad * self.nodes_to_dropout

