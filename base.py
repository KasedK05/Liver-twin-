import numpy as np
from abc import ABC, abstractmethod

class Subsystem(ABC):
    """Abstract base class for all biological subsystems."""

    def __init__(self, params: dict):
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
        """Return state as ordered list for trajectory recording."""
        return list(self.state.values())
