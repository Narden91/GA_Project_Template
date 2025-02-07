import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List


class FitnessVisualizer:
    def __init__(self, max_generations: int, show_plot: bool = True):
        self.max_generations = max_generations
        self.show_plot = show_plot  # Add this flag
        
        plt.style.use('seaborn')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        self.ax.set_xlim(0, max_generations)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.best_line, = self.ax.plot([], [], 'r-', label='Best Fitness', linewidth=2)
        self.avg_line, = self.ax.plot([], [], 'b-', label='Average Fitness', linewidth=2)
        
        self.ax.set_xlabel('Generation', fontsize=12)
        self.ax.set_ylabel('Fitness', fontsize=12)
        self.ax.set_title('Genetic Algorithm Progress', fontsize=14, pad=20)
        self.ax.legend(fontsize=10)
        
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.ax.set_ylim(-1, 1)
    
    def update(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Update the visualization with new fitness values"""
        self.generations.append(generation)
        self.best_fitness.append(best_fitness)
        self.avg_fitness.append(avg_fitness)
        
        self.best_line.set_data(self.generations, self.best_fitness)
        self.avg_line.set_data(self.generations, self.avg_fitness)
        
        max_fitness = max(self.best_fitness + self.avg_fitness, default=1)
        min_fitness = min(self.best_fitness + self.avg_fitness, default=-1)
        padding = (max_fitness - min_fitness) * 0.2 or 0.1
        self.ax.set_ylim(min_fitness - padding, max_fitness + padding)

        if self.show_plot:  # Only show if enabled
            self.fig.canvas.draw()
            plt.pause(0.01)
    
    def save(self, filename: str) -> None:
        """Save the plot to a file with high DPI for better quality"""
        self.fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
