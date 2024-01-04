
import numpy as np


def get_base_np_handler_cls():
    return NumpyHandlerBase

class NumpyHandlerBase:

    @staticmethod
    def convert_neg_values_to_zero(inputs):
        return np.maximum(0, inputs)
    
    @staticmethod
    def do_sum(values_to_sum):
        return np.sum(values_to_sum)

    @staticmethod
    def do_sum_along_axis(values_to_sum, axis_to_perform):
        return np.sum(values_to_sum, axis=axis_to_perform)

    @staticmethod
    def do_abs(values):
        return np.abs(values)

    @staticmethod
    def do_mean(values_to_avg):
        return np.mean(values_to_avg)

    @staticmethod
    def do_log(values):
        return np.log(values)

    @staticmethod
    def argmax(items, axis_to_perform):
        return np.argmax(items, axis=axis_to_perform)

    @staticmethod
    def zeros_like(items):
        return np.zeros_like(items)

    @staticmethod
    def ones_like(items):
        return np.ones_like(items)

    @staticmethod
    def do_clip(items, lower_bound, upper_bound):
        return np.clip(items, lower_bound, upper_bound)