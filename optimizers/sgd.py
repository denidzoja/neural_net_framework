
from np_handlers.base import get_base_np_handler_cls

import functools

def set_pre_post_process_state_updates(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.pre_update_params()
        func(self, *args, **kwargs)
        self.post_update_params()

    return wrapper


class SGDOptimizer:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.np_handler = get_base_np_handler_cls()()
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    @set_pre_post_process_state_updates
    def update_params(self, layer):
        self._initialize_weight_and_bias_momentums(layer)
        if self.momentum:
            weight_updates, bias_updates = self.update_params_with_momentum(layer)
        else:
            weight_updates, bias_updates = self.update_params_without_momentum(layer)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def update_params_with_momentum(self, layer):
        weight_updates = \
            self.momentum * layer.weight_momentums - \
            self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates

        bias_updates = \
            self.momentum * layer.bias_momentums - \
            self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates
        return weight_updates, bias_updates

    def update_params_without_momentum(self, layer):
        weight_updates = -self.current_learning_rate * \
                            layer.dweights
        bias_updates = -self.current_learning_rate * \
                           layer.dbiases
        return weight_updates, bias_updates

    def _initialize_weight_and_bias_momentums(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                self._initialize_layer_weight_momentum(layer)
                self._initialize_layer_bias_momentum(layer)

    def _initialize_layer_weight_momentum(self, layer):
        layer.weight_momentums = self.np_handler.zeros_like(layer.weights)

    def _initialize_layer_bias_momentum(self, layer):
        layer.bias_momentums = self.np_handler.zeros_like(layer.biases)

    def post_update_params(self):
        self.iterations += 1