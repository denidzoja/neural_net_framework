
from abc import ABC, abstractmethod

class LayerBase(ABC):
    
    @abstractmethod
    def forward_pass(self):
        pass

    @abstractmethod
    def backward_pass(self):
        pass
