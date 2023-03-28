# coding: utf-8

from abc import ABC, abstractmethod
import torch.nn as nn


class BatchMetrics(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclasse should implement the reset")

    @abstractmethod
    def update(self, y, yhat):
        raise NotImplementedError("Subclasse should implement the reset")

    @abstractmethod
    def value(self):
        raise NotImplementedError("Subclasse should implement the reset")


class CE(BatchMetrics):
    def __init__(self):
        self.reset()
        self.loss = nn.CrossEntropyLoss(reduction="sum")

    def reset(self):
        self._n_samples = 0
        self._cum_value = 0

    def update(self, input_tensor, target_tensor):
        """
        input_tensor (B, C)
        target_tensor (B, )
        """
        self._n_samples += input_tensor.shape[0]
        self._cum_value += self.loss(input_tensor, target_tensor).item()

    def value(self):
        return self._cum_value / self._n_samples


class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self._n_samples = 0
        self._cum_value = 0

    def update(self, input_tensor, target_tensor):
        """
        input_tensor (B, C)
        target_tensor (B, )
        """
        self._n_samples += input_tensor.shape[0]
        predicted_tensor = input_tensor.argmax(dim=1)
        self._cum_value += (predicted_tensor == target_tensor).sum().item()

    def value(self):
        return self._cum_value / self._n_samples
