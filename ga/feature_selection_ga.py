import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ga.genetic_algorithm import GeneticAlgorithm, EncodingType, SelectionType, CrossoverType, MutationType
from sklearn.ensemble import RandomForestClassifier
from rich.console import Console
from rich.panel import Panel
from sklearn.feature_selection import mutual_info_classif


console = Console()


class FeatureSelectionGA(GeneticAlgorithm):
    def __init__(self, X_train, y_train, model=None, pop_size=50, mutation_rate=0.1, 
                 crossover_rate=0.8, elitism=True, seed=42, verbose=False, fitness_type="wrapper"):
        """
        Initialize GA for feature selection.
        
        Parameters:
          - fitness_type: "wrapper" (default) uses model accuracy, 
                          "filter" uses correlation and mutual information scores.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_features = X_train.shape[1]
        self.verbose = verbose
        self.seed = seed
        self.fitness_type = fitness_type  # New field for selecting fitness type
        
        # Default classifier if none is provided
        self.model = model if model else DecisionTreeClassifier(random_state=seed)
        
        # Pass our evaluation method to the GeneticAlgorithm
        super().__init__(
            fitness_func=self.evaluate_fitness,
            pop_size=pop_size,
            chromosome_length=self.num_features,  # One gene per feature
            gene_bounds=(0, 1),  # Binary encoding for feature selection
            encoding_type=EncodingType.BINARY,
            selection_type=SelectionType.TOURNAMENT,  
            crossover_type=CrossoverType.ONE_POINT,  
            mutation_type=MutationType.BIT_FLIP,  
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism=elitism
        )

        if self.verbose:
            console.print(Panel.fit(
                f"[bold cyan]Feature Selection GA Initialized[/bold cyan]\n"
                f"Population Size: {pop_size}\n"
                f"Mutation Rate: {mutation_rate}\n"
                f"Crossover Rate: {crossover_rate}\n"
                f"Elitism: {elitism}\n"
                f"Number of Features: {self.num_features}\n"
                f"Fitness Type: {self.fitness_type}"
            ))

    def _initialize_population(self) -> np.ndarray:
        """Override GeneticAlgorithm's population initialization to ensure correct shape."""
        population = np.random.randint(2, size=(self.pop_size, self.num_features))
        
        # Ensure at least one feature is selected per chromosome
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                population[i, np.random.randint(0, self.num_features)] = 1
        
        return population

    def _mutation_bit_flip(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit flip mutation ensuring strictly binary encoding."""
        mask = np.random.random(len(chromosome)) < self.mutation_rate
        chromosome[mask] = 1 - chromosome[mask]  # Flip 0 to 1 and vice versa
        return chromosome

    def evaluate_fitness(self, chromosome) -> float:
        """
        Dispatch fitness evaluation based on fitness_type.
        """
        if self.fitness_type == "wrapper":
            return self.evaluate_wrapper_fitness(chromosome)
        elif self.fitness_type == "filter":
            return self.evaluate_filter_fitness(chromosome)
        else:
            raise ValueError(f"Unknown fitness_type: {self.fitness_type}")

    def evaluate_wrapper_fitness(self, chromosome) -> float:
        """Wrapper fitness: accuracy with penalty for number of features."""
        chromosome = np.round(chromosome).astype(int)  # Ensure strict binary (0/1)
        selected_features = np.where(chromosome == 1)[0]

        # If no features are selected, return worst fitness
        if len(selected_features) == 0:
            return 0

        X_selected = self.X_train[:, selected_features]

        # Train model & compute accuracy (no cross-validation)
        self.model.fit(X_selected, self.y_train)
        accuracy = self.model.score(X_selected, self.y_train)

        # Apply a trade-off between accuracy and feature count
        penalty = len(selected_features) / self.num_features
        fitness = accuracy - 0.1 * penalty

        return fitness


    def evaluate_filter_fitness(self, chromosome) -> float:
        """
        Filter-based fitness:
          - Computes the absolute Pearson correlation (linear) for each selected feature.
          - Computes the mutual information (non-linear) for each selected feature.
          - Combines the average of these scores and applies a penalty for selecting too many features.
        """
        chromosome = np.round(chromosome).astype(int)  # Ensure binary encoding
        selected_features = np.where(chromosome == 1)[0]

        # If no features are selected, return worst fitness
        if len(selected_features) == 0:
            return 0

        X_selected = self.X_train[:, selected_features]

        # Compute linear (Pearson) scores for each feature
        linear_scores = []
        for i in range(X_selected.shape[1]):
            feat = X_selected[:, i]
            # Avoid division by zero if feature has no variance
            if np.std(feat) == 0:
                linear_scores.append(0)
            else:
                corr = np.corrcoef(feat, self.y_train)[0, 1]
                linear_scores.append(abs(corr))
        linear_mean = np.mean(linear_scores)

        # Compute non-linear (mutual information) scores for each feature
        mi_scores = mutual_info_classif(X_selected, self.y_train, random_state=self.seed)
        mi_mean = np.mean(mi_scores)

        # Combine the two measures (here a simple sum; adjust weights if needed)
        combined_score = linear_mean + mi_mean

        # Penalize larger feature subsets (similar to wrapper fitness)
        penalty = len(selected_features) / self.num_features
        fitness = combined_score - 0.1 * penalty

        return fitness

    def evolve(self, run, logger=None):
        """Perform one generation of evolution, overriding GeneticAlgorithm's evolve method."""
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
        selected = self._selection_tournament()

        # Create new population through crossover
        new_population = np.zeros_like(self.population)
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size:
                new_population[i], new_population[i+1] = self._crossover_one_point(
                    selected[i], selected[i+1]
                )
            else:
                new_population[i] = selected[i]

        # Mutation
        for i in range(self.pop_size):
            new_population[i] = self._mutation_bit_flip(new_population[i])

        # Elitism: preserve best individual
        if self.elitism:
            best_idx = np.argmax(self.fitness_scores)
            worst_idx = np.argmin(self.fitness_scores)
            new_population[worst_idx] = self.population[best_idx]

        self.population = new_population
        return best_fitness, avg_fitness
