from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional

class Encoding(ABC):
    @abstractmethod
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        """Convert phenotype to genotype"""
        pass
    
    @abstractmethod
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        """Convert genotype to phenotype"""
        pass
    
    @abstractmethod
    def initialize_population(self, pop_size: int, chromosome_length: int, bounds: Tuple[float, float]) -> np.ndarray:
        """Initialize a population with specific encoding"""
        pass

class FloatEncoding(Encoding):
    def __init__(self, precision: int = 6):
        self.precision = precision
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        """Float encoding is direct, so just return the array"""
        return phenotype.copy()
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        """Float decoding is direct, so just return the array"""
        return genotype.copy()
    
    def initialize_population(self, pop_size: int, chromosome_length: int, bounds: Tuple[float, float]) -> np.ndarray:
        """Initialize population with float values within bounds"""
        return np.random.uniform(
            bounds[0],
            bounds[1],
            (pop_size, chromosome_length)
        )

class BinaryEncoding(Encoding):
    def __init__(self, bits_per_param: int = 16):
        self.bits_per_param = bits_per_param
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        """Convert float values to binary representation"""
        binary = []
        for value in phenotype:
            # Convert to binary string and remove '0b' prefix
            bin_str = bin(int(value * (2**self.bits_per_param)))[2:]
            # Pad with zeros if necessary
            bin_str = bin_str.zfill(self.bits_per_param)
            # Convert to array of integers
            binary.extend([int(bit) for bit in bin_str])
        return np.array(binary)
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        """Convert binary representation back to float values"""
        n_params = len(genotype) // self.bits_per_param
        phenotype = np.zeros(n_params)
        
        for i in range(n_params):
            # Get binary string for current parameter
            start = i * self.bits_per_param
            end = start + self.bits_per_param
            bin_str = ''.join(map(str, genotype[start:end]))
            # Convert to float
            value = int(bin_str, 2) / (2**self.bits_per_param)
            phenotype[i] = value
        
        return phenotype
    
    def initialize_population(self, pop_size: int, chromosome_length: int, bounds: Tuple[float, float]) -> np.ndarray:
        """Initialize population with binary values"""
        print(f"[DEBUG] BinaryEncoding: Pop Size = {pop_size}, Chromosome Length = {chromosome_length}")
        total_bits = chromosome_length * self.bits_per_param
        return np.random.randint(2, size=(pop_size, total_bits))

class GrayBinaryEncoding(BinaryEncoding):
    def __init__(self, bits_per_param: int = 16):
        super().__init__(bits_per_param)
    
    def _binary_to_gray(self, binary: np.ndarray) -> np.ndarray:
        """Convert binary to Gray code"""
        gray = np.zeros_like(binary)
        gray[0] = binary[0]
        gray[1:] = binary[:-1] ^ binary[1:]
        return gray
    
    def _gray_to_binary(self, gray: np.ndarray) -> np.ndarray:
        """Convert Gray code to binary"""
        binary = np.zeros_like(gray)
        binary[0] = gray[0]
        for i in range(1, len(gray)):
            binary[i] = binary[i-1] ^ gray[i]
        return binary
    
    def encode(self, phenotype: np.ndarray) -> np.ndarray:
        """Convert float values to Gray-coded binary representation"""
        binary = super().encode(phenotype)
        return self._binary_to_gray(binary)
    
    def decode(self, genotype: np.ndarray) -> np.ndarray:
        """Convert Gray-coded binary representation back to float values"""
        binary = self._gray_to_binary(genotype)
        return super().decode(binary)