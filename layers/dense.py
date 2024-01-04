
from np_handlers.np_dense import get_dense_layer_np_handler_cls

from .base import LayerBase

class DenseLayer(LayerBase):

    def __init__(self, num_inputs, num_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.np_handler = get_dense_layer_np_handler_cls()(self)

        self.weights = self.np_handler.initialize_weights(num_inputs, num_neurons)
        self.biases = self.np_handler.initialize_biases(num_neurons)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward_pass(self, inputs, training):
        self.inputs = inputs
        self.output = self.np_handler.forward_pass(inputs, self.weights, self.biases)

    def backward_pass(self, input_grad):
        self.dweights = self.np_handler.calculate_dweights(self.inputs, input_grad)
        self.dinputs = self.np_handler.calculate_dinputs(input_grad, self.weights)
        self.dbiases = self.np_handler.calculate_dbiases(input_grad)

        self.dweights += self._calculate_weight_L1_reg()
        self.dweights += self._calculate_weight_L2_reg()
        self.dbiases += self._calculate_bias_L1_reg()
        self.dbiases += self._calculate_bias_L2_reg()

    def _calculate_weight_L1_reg(self):
        partial_der_of_abs_func = self.np_handler.ones_like(self.weights)
        partial_der_of_abs_func[self.weights < 0] = -1
        return self.weight_regularizer_l1 * partial_der_of_abs_func

    def _calculate_weight_L2_reg(self):
        weight_regularizer_L2 = 2 * self.weight_regularizer_l2 * self.weights
        return weight_regularizer_L2

    def _calculate_bias_L1_reg(self):
        partial_der_of_abs_func = self.np_handler.ones_like(self.biases)
        partial_der_of_abs_func[self.biases < 0] = -1
        return self.bias_regularizer_l1 * partial_der_of_abs_func

    def _calculate_bias_L2_reg(self):
        bias_L2_reg = 2 * self.bias_regularizer_l2 * self.biases
        return bias_L2_reg
