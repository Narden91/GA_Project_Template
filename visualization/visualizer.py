import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List


class FitnessVisualizer:
    def __init__(self, max_generations: int):
        self.max_generations = max_generations
        
        # Setup the plot with a larger figure size and better margins
        plt.style.use('seaborn')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # Initialize plot configuration
        self.ax.set_xlim(0, max_generations)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Initialize empty lines with increased line width
        self.best_line, = self.ax.plot([], [], 'r-', label='Best Fitness', linewidth=2)
        self.avg_line, = self.ax.plot([], [], 'b-', label='Average Fitness', linewidth=2)
        
        # Setup labels and title with larger font sizes
        self.ax.set_xlabel('Generation', fontsize=12)
        self.ax.set_ylabel('Fitness', fontsize=12)
        self.ax.set_title('Genetic Algorithm Progress', fontsize=14, pad=20)
        self.ax.legend(fontsize=10)
        
        # Data storage
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        
        # Initial y-axis limits with padding
        self.ax.set_ylim(-1, 1)  # Will be adjusted dynamically
        
    def update(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Update the visualization with new fitness values"""
        self.generations.append(generation)
        self.best_fitness.append(best_fitness)
        self.avg_fitness.append(avg_fitness)
        
        # Update line data
        self.best_line.set_data(self.generations, self.best_fitness)
        self.avg_line.set_data(self.generations, self.avg_fitness)
        
        # Calculate y-axis limits with extra padding
        if len(self.best_fitness) > 0:
            max_fitness = max(max(self.best_fitness), max(self.avg_fitness))
            min_fitness = min(min(self.best_fitness), min(self.avg_fitness))
            
            # Add padding to prevent cropping
            y_range = max_fitness - min_fitness
            padding = y_range * 0.2  # 20% padding
            
            if padding == 0:  # Handle case where all values are the same
                padding = abs(max_fitness) * 0.1 if max_fitness != 0 else 0.1
                
            self.ax.set_ylim(
                min_fitness - padding,
                max_fitness + padding
            )
        
        # Draw the plot
        self.fig.canvas.draw()
        plt.pause(0.01)  # Small pause to allow for real-time updates
        
    def save(self, filename: str) -> None:
        """Save the plot to a file with high DPI for better quality"""
        # Ensure the figure is properly sized and scaled
        self.fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')