from abc import ABC, abstractmethod

class MatrixCompletion(ABC):

    @abstractmethod
    def fit(self, Y, mask):
        pass

    @abstractmethod
    def predict(self):
        pass