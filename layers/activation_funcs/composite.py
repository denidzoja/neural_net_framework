
from layers.base import LayerBase

from np_handlers.np_softmax import get_softmax_layer_np_handler_cls

class SoftmaxActivationCategoricalCrossEntropyLoss(LayerBase):

    def __init__(self):
        self.np_handler = get_softmax_layer_np_handler_cls()(self)

    def forward_pass(self):
        pass

    def backward_pass(self, input_grad, true_values):
        num_samples = len(input_grad)
        true_values = self.np_handler.convert_one_hot_to_discrete_labels(true_values)
        self.dinputs = input_grad.copy()
        self.dinputs[range(num_samples), true_values] -= 1
        self.dinputs = self.dinputs / num_samples
