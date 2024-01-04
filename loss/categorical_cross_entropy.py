
from .loss_base import LossBase


class CategoricalCrossEntropyLoss(LossBase):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, true_values):

        samples = len(predictions)
        predictions_clipped = self.np_handler.do_clip(predictions, 1e-7, 1 - 1e-7)

        if self.is_categorical_label(true_values):
            correct_confidences = predictions_clipped[
                range(samples),
                true_values
            ]
        elif self.is_one_hot_label(true_values):
            correct_confidences = self.np_handler.do_sum_along_axis(predictions_clipped * true_values,
                                                                    axis_to_perform=1)
        negative_log_likelihoods = -self.np_handler.do_log(correct_confidences)
        return negative_log_likelihoods

    def is_categorical_label(self, input):
        return len(input.shape) == 1

    def is_one_hot_label(self, input):
        return len(input.shape) == 2
