
from np_handlers.base import get_base_np_handler_cls

class LossBase:
    def __init__(self):
        self.np_handler = get_base_np_handler_cls()()

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            regularization_loss += self._calculate_weight_L1_reg_loss(layer)
            regularization_loss += self._calculate_weight_L2_reg_loss(layer)
            regularization_loss += self._calculate_bias_L1_reg_loss(layer)
            regularization_loss += self._calculate_bias_L2_reg_loss(layer)
        return regularization_loss

    def initialize_loss_func(self, trainable_layers):
        self._set_trainable_layers(trainable_layers)

    def calculate(self, output, true_values, include_regularization=False):
        sample_losses = self.forward(output, true_values)
        data_loss = self.np_handler.do_mean(sample_losses)

        self.accumulated_sum += self.np_handler.do_sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def reset_state(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def _set_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def _calculate_weight_L1_reg_loss(self, layer):
        abs_weights = self.np_handler.do_abs(layer.weights)
        sum_of_abs_weights = self.np_handler.do_sum(abs_weights)
        loss = layer.weight_regularizer_l1 * sum_of_abs_weights
        return loss

    def _calculate_weight_L2_reg_loss(self, layer):
        squared_weights = layer.weights * layer.weights
        sum_of_sq_weights = self.np_handler.do_sum(squared_weights)
        loss = layer.weight_regularizer_l2 * sum_of_sq_weights
        return loss

    def _calculate_bias_L1_reg_loss(self, layer):
        abs_biases = self.np_handler.do_abs(layer.biases)
        sum_of_abs_biases = self.np_handler.do_sum(abs_biases)
        loss = layer.bias_regularizer_l1 * sum_of_abs_biases
        return loss

    def _calculate_bias_L2_reg_loss(self, layer):
        squared_biases = layer.biases * layer.biases
        sum_of_sq_biases = self.np_handler.do_sum(squared_biases)
        loss = layer.bias_regularizer_l2 * sum_of_sq_biases
        return loss
