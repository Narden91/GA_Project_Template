import numpy as np
from ga.genetic_algorithm import GeneticAlgorithm, EncodingType
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


class FeatureSelectionGA(GeneticAlgorithm):
    def __init__(
        self,
        X_train,
        y_train,
        model,
        pop_size=50,
        chromosome_length=None,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism=True,
    ):
        """Initialize GA for feature selection."""
        self.X_train = X_train
        self.y_train = y_train
        self.model = clone(model)  # Clone the provided ML model
        self.num_features = X_train.shape[1]
        chromosome_length = chromosome_length or self.num_features

        super().__init__(
            fitness_func=self.evaluate_fitness,
            pop_size=pop_size,
            chromosome_length=chromosome_length,
            gene_bounds=(0, 1),  # Binary representation for feature selection
            encoding_type=EncodingType.BINARY,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism=elitism,
        )

    def evaluate_fitness(self, chromosome):
        """Evaluate feature subset based on cross-validated accuracy."""
        selected_features = np.where(chromosome == 1)[0]
        if len(selected_features) == 0:
            return 0 

        X_selected = self.X_train[:, selected_features]
        score = cross_val_score(self.model, X_selected, self.y_train, cv=3, scoring="accuracy").mean()
        
        # Return fitness as accuracy minus a penalty for feature count
        penalty = len(selected_features) / self.num_features  # Reduce feature count
        return score - 0.1 * penalty  # Trade-off between accuracy and feature reduction
