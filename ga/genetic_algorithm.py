import numpy as np
from typing import Callable, List, Tuple, Optional
from enum import Enum
from .encoding import Encoding, FloatEncoding, BinaryEncoding, GrayBinaryEncoding

class SelectionType(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"

class CrossoverType(Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"

class MutationType(Enum):
    GAUSSIAN = "gaussian"
    RANDOM_RESET = "random_reset"
    SWAP = "swap"
    BIT_FLIP = "bit_flip"

class EncodingType(Enum):
    FLOAT = "float"
    BINARY = "binary"
    GRAY_BINARY = "gray_binary"

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_func: Callable,
        pop_size: int,
        chromosome_length: int,
        gene_bounds: Tuple[float, float],
        encoding_type: EncodingType = EncodingType.FLOAT,
        encoding_params: dict = None,
        selection_type: SelectionType = SelectionType.TOURNAMENT,
        crossover_type: CrossoverType = CrossoverType.TWO_POINT,
        mutation_type: MutationType = MutationType.GAUSSIAN,
        tournament_size: int = 3,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism: bool = True
    ):
        
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.gene_bounds = gene_bounds
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
    
        # Set up encoding
        self._setup_encoding(encoding_type, encoding_params or {})
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(pop_size)
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def _setup_encoding(self, encoding_type: EncodingType, params: dict):
        """Initialize the appropriate encoding scheme"""
        if encoding_type == EncodingType.FLOAT:
            self.encoding = FloatEncoding(
                precision=params.get('precision', 6)
            )
        elif encoding_type == EncodingType.BINARY:
            self.encoding = BinaryEncoding(
                bits_per_param=params.get('bits_per_param', 16)
            )
        elif encoding_type == EncodingType.GRAY_BINARY:
            self.encoding = GrayBinaryEncoding(
                bits_per_param=params.get('bits_per_param', 16)
            )
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize population using the selected encoding"""
        return self.encoding.initialize_population(
            self.pop_size,
            self.chromosome_length,
            self.gene_bounds
        )
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for entire population"""
        for i in range(self.pop_size):
            # Decode chromosome before fitness evaluation
            phenotype = self.encoding.decode(self.population[i])
            # Scale phenotype to bounds if using binary encoding
            if isinstance(self.encoding, (BinaryEncoding, GrayBinaryEncoding)):
                phenotype = phenotype * (self.gene_bounds[1] - self.gene_bounds[0]) + self.gene_bounds[0]
            self.fitness_scores[i] = self.fitness_func(phenotype)

    def _mutation_bit_flip(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit flip mutation for binary encodings"""
        mask = np.random.random(len(chromosome)) < self.mutation_rate
        chromosome[mask] = 1 - chromosome[mask]
        return chromosome
    
    def _selection_tournament(self) -> np.ndarray:
        """Tournament selection"""
        selected = np.zeros_like(self.population)
        for i in range(self.pop_size):
            tournament_idx = np.random.choice(self.pop_size, self.tournament_size)
            tournament_fitness = self.fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected[i] = self.population[winner_idx]
        return selected
    
    def _selection_roulette(self) -> np.ndarray:
        """Roulette wheel selection"""
        fitness_sum = np.sum(self.fitness_scores)
        probs = self.fitness_scores / fitness_sum
        selected_idx = np.random.choice(
            self.pop_size,
            size=self.pop_size,
            p=probs
        )
        return self.population[selected_idx]
    
    def _selection_rank(self) -> np.ndarray:
        """Rank-based selection"""
        ranks = np.argsort(np.argsort(self.fitness_scores))
        probs = (ranks + 1) / np.sum(ranks + 1)
        selected_idx = np.random.choice(
            self.pop_size,
            size=self.pop_size,
            p=probs
        )
        return self.population[selected_idx]
    
    def _crossover_one_point(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-point crossover"""
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.chromosome_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2
    
    def _crossover_two_point(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover"""
        if np.random.random() < self.crossover_rate:
            points = sorted(np.random.choice(self.chromosome_length, 2, replace=False))
            child1 = np.concatenate([
                parent1[:points[0]],
                parent2[points[0]:points[1]],
                parent1[points[1]:]
            ])
            child2 = np.concatenate([
                parent2[:points[0]],
                parent1[points[0]:points[1]],
                parent2[points[1]:]
            ])
            return child1, child2
        return parent1, parent2
    
    def _crossover_uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover"""
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(self.chromosome_length) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1, parent2
    
    def _mutation_gaussian(self, chromosome: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mask = np.random.random(self.chromosome_length) < self.mutation_rate
        mutation = np.random.normal(0, 0.1, self.chromosome_length)
        chromosome[mask] += mutation[mask]
        return np.clip(chromosome, self.gene_bounds[0], self.gene_bounds[1])
    
    def _mutation_random_reset(self, chromosome):
        """Performs random reset mutation on a chromosome."""
        chromosome = np.array(chromosome)  # Ensure array type
        mask = np.random.rand(chromosome.shape[0]) < self.mutation_rate

        if mask.shape[0] != chromosome.shape[0]:
            raise ValueError(f"Mismatch in mutation: chromosome size {chromosome.shape[0]}, mask size {mask.shape[0]}")

        chromosome[mask] = np.random.uniform(
            self.gene_bounds[0], self.gene_bounds[1], size=np.sum(mask)
        )
        return chromosome
    
    def _mutation_swap(self, chromosome: np.ndarray) -> np.ndarray:
        """Swap mutation"""
        if np.random.random() < self.mutation_rate:
            idx1, idx2 = np.random.choice(self.chromosome_length, 2, replace=False)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome
    
    def evolve(self, run, logger=None):
        """Perform one generation of evolution."""
        self._evaluate_population()

        # Store fitness statistics
        best_fitness = np.max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Log the data if a logger is provided
        if logger:
            logger.log_generation(run, len(self.best_fitness_history) - 1, avg_fitness, best_fitness)

        # Selection
        if self.selection_type == SelectionType.TOURNAMENT:
            selected = self._selection_tournament()
        elif self.selection_type == SelectionType.ROULETTE:
            selected = self._selection_roulette()
        else:
            selected = self._selection_rank()

        # Create new population through crossover
        new_population = np.zeros_like(self.population)
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size:
                if self.crossover_type == CrossoverType.ONE_POINT:
                    new_population[i], new_population[i+1] = self._crossover_one_point(
                        selected[i], selected[i+1]
                    )
                elif self.crossover_type == CrossoverType.TWO_POINT:
                    new_population[i], new_population[i+1] = self._crossover_two_point(
                        selected[i], selected[i+1]
                    )
                else:
                    new_population[i], new_population[i+1] = self._crossover_uniform(
                        selected[i], selected[i+1]
                    )
            else:
                new_population[i] = selected[i]

        # Mutation
        for i in range(self.pop_size):
            if isinstance(self.encoding, (BinaryEncoding, GrayBinaryEncoding)):
                if self.mutation_type == MutationType.BIT_FLIP:
                    new_population[i] = self._mutation_bit_flip(new_population[i])
                else:
                    new_population[i] = self._mutation_random_reset(new_population[i])
            else:
                if self.mutation_type == MutationType.GAUSSIAN:
                    new_population[i] = self._mutation_gaussian(new_population[i])
                elif self.mutation_type == MutationType.RANDOM_RESET:
                    new_population[i] = self._mutation_random_reset(new_population[i])
                else:
                    new_population[i] = self._mutation_swap(new_population[i])

        # Elitism: preserve best individual
        if self.elitism:
            best_idx = np.argmax(self.fitness_scores)
            worst_idx = np.argmin(self.fitness_scores)
            new_population[worst_idx] = self.population[best_idx]

        self.population = new_population
        
            
        return best_fitness, avg_fitness