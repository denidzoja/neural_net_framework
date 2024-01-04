
from .base import AccuracyBase


class CategoricalAccuracy(AccuracyBase):

    def __init__(self, binary=False):
        super().__init__()
        self.binary = binary

    def compare_predictions(self, predictions, true_values):
        if not self.binary and len(true_values.shape) == 2:
            true_values = self.np_handler.argmax(true_values, axis_to_perform=1)
        return predictions == true_values
