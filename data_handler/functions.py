import numpy as np
from typing import Callable, Dict
from enum import Enum


class OptimizationFunction(Enum):
    ACKLEY = "ackley"
    ROSENBROCK = "rosenbrock"
    SCHWEFEL = "schwefel"


class OptimizationProblem:
    def __init__(self, function_type: OptimizationFunction):
        self.function_type = function_type
        self._setup_function()
    
    def _setup_function(self):
        """Set up the optimization function and its parameters"""
        if self.function_type == OptimizationFunction.ACKLEY:
            self.function = self._ackley
            self.bounds = (-32.768, 32.768)
            self.optimal_value = 0
            self.description = "Ackley function: Multi-modal test function with many local minima"
        elif self.function_type == OptimizationFunction.ROSENBROCK:
            self.function = self._rosenbrock
            self.bounds = (-2.048, 2.048)
            self.optimal_value = 0
            self.description = "Rosenbrock function: Valley-shaped function with global minimum inside a narrow parabolic valley"
        elif self.function_type == OptimizationFunction.SCHWEFEL:
            self.function = self._schwefel
            self.bounds = (-500, 500)
            self.optimal_value = 0
            self.description = "Schwefel function: Deceptive function where global minimum is far from the next best local minima"
    
    def _ackley(self, x: np.ndarray) -> float:
        """
        Ackley function implementation
        Global minimum: f(0,0,...,0) = 0
        """
        n = len(x)
        sum_sq_term = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        cos_term = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
        return -(sum_sq_term + cos_term + 20 + np.exp(1))
    
    def _rosenbrock(self, x: np.ndarray) -> float:
        """
        Rosenbrock function implementation
        Global minimum: f(1,1,...,1) = 0
        """
        sum_term = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        return -sum_term
    
    def _schwefel(self, x: np.ndarray) -> float:
        """
        Schwefel function implementation
        Global minimum: f(420.9687,...,420.9687) = 0
        """
        n = len(x)
        sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return -(418.9829 * n - sum_term)
    
    def get_function_info(self) -> Dict:
        """Return information about the optimization function"""
        return {
            "name": self.function_type.value,
            "bounds": self.bounds,
            "optimal_value": self.optimal_value,
            "description": self.description
        }
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the optimization function"""
        return self.function(x)