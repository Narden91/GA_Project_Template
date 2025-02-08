from rich.console import Console
from rich.table import Table
import yaml
from data_handler.functions import OptimizationProblem
from ga.genetic_algorithm import GeneticAlgorithm


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