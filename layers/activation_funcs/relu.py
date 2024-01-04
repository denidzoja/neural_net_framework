
from np_handlers.base import get_base_np_handler_cls

from ..base import LayerBase


class Activation_ReLU(LayerBase):
    def __init__(self):
        self.np_handler = get_base_np_handler_cls()()

    def forward_pass(self, inputs, training):
        self.inputs = inputs
        self.output = self.np_handler.convert_neg_values_to_zero(inputs)

    def backward_pass(self, input_grad):
        self.dinputs = input_grad.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs
