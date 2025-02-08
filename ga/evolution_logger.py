import pandas as pd
import os

class EvolutionLogger:
    def __init__(self, output_dir="results", filename="evolution_data.csv"):
        """Initialize the logger with an empty DataFrame."""
        self.output_dir = output_dir
        self.filename = filename
        self.data = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def log_generation(self, run, generation, avg_fitness, best_fitness):
        """Store evolution data for each generation."""
        self.data.append({
            "Run": run,
            "Generation": generation,
            "Average Fitness": avg_fitness,
            "Best Fitness": best_fitness
        })

    def save(self):
        """Save the logged data to a CSV file."""
        df = pd.DataFrame(self.data)
        output_path = os.path.join(self.output_dir, self.filename)
        df.to_csv(output_path, index=False)