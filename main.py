import os
import sys

sys.dont_write_bytecode = True

from ga.evolution_logger import EvolutionLogger
from ga import GeneticAlgorithm, SelectionType, CrossoverType, MutationType
from visualization import FitnessVisualizer
from data_handler import OptimizationProblem, OptimizationFunction
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
import numpy as np
import yaml
import time


# Initialize Rich console
console = Console()


def load_config(config_path: str = "config/parameters.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {str(e)}[/bold red]")
        raise

def print_problem_info(problem: OptimizationProblem) -> None:
    """Print optimization problem information"""
    info = problem.get_function_info()
    table = Table(title="Optimization Problem Information", show_header=True, header_style="bold magenta")
    
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Function", info["name"].capitalize())
    table.add_row("Description", info["description"])
    table.add_row("Bounds", f"{info['bounds']}")
    table.add_row("Optimal Value", str(info["optimal_value"]))
    
    console.print(table)

def print_ga_config(ga: GeneticAlgorithm, config: dict) -> None:
    """Print GA configuration"""
    table = Table(title="Genetic Algorithm Configuration", show_header=True, header_style="bold magenta")
    
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Population Size", str(ga.pop_size))
    table.add_row("Chromosome Length", str(ga.chromosome_length))
    table.add_row("Selection Type", ga.selection_type.value)
    table.add_row("Crossover Type", ga.crossover_type.value)
    table.add_row("Mutation Type", ga.mutation_type.value)
    table.add_row("Tournament Size", str(ga.tournament_size))
    table.add_row("Mutation Rate", f"{ga.mutation_rate:.3f}")
    table.add_row("Crossover Rate", f"{ga.crossover_rate:.3f}")
    table.add_row("Elitism", str(ga.elitism))
    
    console.print(table)


def main():
    # Load configuration
    config = load_config()
    
    # Print header
    console.print(Panel.fit(
        "[bold blue]Genetic Algorithm Optimization[/bold blue]\n"
        "[italic]Configuration loaded from YAML file[/italic]"
    ))
    
    # Initialize optimization problem
    problem = OptimizationProblem(
        OptimizationFunction(config['optimization']['function'])
    )
    
    # Print problem information
    print_problem_info(problem)
    
    # Setup progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        
        # Add evolution task
        evolution_task = progress.add_task(
            "[cyan]Evolving...",
            total=config['genetic_algorithm']['max_generations']
        )
        
        # Initialize evolution logger
        logger = EvolutionLogger()
        
                # Track best fitness across all runs
        best_fitness_overall = float('-inf')
        best_solution_overall = None

        for run in range(config['genetic_algorithm']['runs']):
            console.print(f"\n[bold blue]Run {run + 1}/{config['genetic_algorithm']['runs']}[/bold blue]")

            ga = GeneticAlgorithm(
                fitness_func=problem.evaluate,
                pop_size=config['genetic_algorithm']['population_size'],
                chromosome_length=config['genetic_algorithm']['chromosome_length'],
                gene_bounds=problem.bounds,
                selection_type=SelectionType(config['operators']['selection']),
                crossover_type=CrossoverType(config['operators']['crossover']),
                mutation_type=MutationType(config['operators']['mutation']),
                tournament_size=config['genetic_algorithm']['tournament_size'],
                mutation_rate=config['genetic_algorithm']['mutation_rate'],
                crossover_rate=config['genetic_algorithm']['crossover_rate'],
                elitism=config['genetic_algorithm']['elitism']
            )

            # Print GA configuration for first run if verbose
            if run == 0 and config['settings']['verbose']:
                print_ga_config(ga, config)

            # Initialize best fitness for the current run
            best_solution = None
            best_overall_fitness = float('-inf')

            # Initialize visualizer for each run
            visualizer = None
            if config['visualization']['show_plot'] or config['visualization']['save_plot']:
                visualizer = FitnessVisualizer(config['genetic_algorithm']['max_generations'], show_plot=config['visualization']['show_plot'])

            for generation in range(config['genetic_algorithm']['max_generations']):
                # Perform one generation of evolution
                best_fitness, avg_fitness = ga.evolve(run, logger)

                # Track best solution and fitness for the current run
                if best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_idx = np.argmax(ga.fitness_scores)
                    best_solution = ga.population[best_idx].copy()

                # Update visualization if enabled
                if config['visualization']['show_plot'] or config['visualization']['save_plot']:
                    visualizer.update(generation, best_fitness, avg_fitness)

                # Update progress
                progress.update(evolution_task, advance=1)
                console.print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}", end="\r")

                time.sleep(config['visualization']['update_interval'])

            # Check if this run produced the best fitness overall
            if best_overall_fitness > best_fitness_overall:
                best_fitness_overall = best_overall_fitness
                best_solution_overall = best_solution

            # Save fitness plot if enabled
            if config['visualization']['save_plot']:
                if not os.path.exists(config['visualization']['output_folder']):
                    os.makedirs(config['visualization']['output_folder'])
                
                fitness_filename = os.path.join(
                    config['visualization']['output_folder'], 
                    f"{config['visualization']['output_filename'].split('.')[0]}_run_{run + 1}.png"
                )

                visualizer.save(fitness_filename)
                console.print(f"\n[italic]Fitness evolution plot for run {run + 1} saved as '{fitness_filename}'[/italic]") if config['settings']['verbose'] else None

        # Save evolution data to CSV
        logger.save()

        # Print final results
        console.print("\n\n[bold green]Optimization Complete![/bold green]")

        results_table = Table(title="Final Results", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Best Fitness (Overall)", f"{best_fitness_overall:.6f}")
        results_table.add_row("Best Solution (Overall)", str(best_solution_overall))
        results_table.add_row("Distance from Optimum", f"{abs(best_fitness_overall - problem.optimal_value):.6f}")

        console.print(results_table)


if __name__ == "__main__":
    main()