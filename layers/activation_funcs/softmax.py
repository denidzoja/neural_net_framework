
from np_handlers.np_softmax import get_softmax_layer_np_handler_cls

from ..base import LayerBase


class SoftmaxActivation(LayerBase):

    def __init__(self):
        self.np_handler = get_softmax_layer_np_handler_cls()(self)

    def forward_pass(self, inputs, training):
        self.inputs = inputs
        probabilities = self.np_handler.get_softmax_probabilities_from_inputs(inputs)
        self.output = probabilities

    def backward_pass(self):
        pass

    def predictions(self, outputs):
        return self.np_handler.argmax(outputs, axis_to_perform=1)

