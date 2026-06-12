from abc import ABC, abstractmethod
import numpy as np

class Subsystem(ABC):
    def __init__(self, params):
        self.params = params
        self.state = self.initialize_state(params)

    @abstractmethod
    def initialize_state(self, params):
        pass

    @abstractmethod
    def update(self, dt, coupling_inputs, perturbation_deltas):
        pass

    @abstractmethod
    def get_health_score(self):
        pass

    def get_state_vector(self):
        return np.array(list(self.state.values()), dtype=float)
