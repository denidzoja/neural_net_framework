
from np_handlers.base import get_base_np_handler_cls

class AccuracyBase:

    def __init__(self):
        self.np_handler = get_base_np_handler_cls()()

    def calculate(self, predictions, true_values):

        comparisons = self.compare_predictions(predictions, true_values)
        accuracy = self.np_handler.do_mean(comparisons)

        self.accumulated_sum += self.np_handler.do_sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def reset_state(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


