# Genetic Algorithm Project

This project implements a flexible and modular genetic algorithm framework for solving complex optimization problems. It features real-time visualization, configurable parameters, and multiple optimization functions.

## Overview

The project provides a comprehensive genetic algorithm implementation with customizable operators, real-time fitness visualization, and a selection of challenging optimization functions including Ackley, Rosenbrock, and Schwefel functions. All parameters are easily configurable through a YAML file, making it simple to experiment with different settings and problems.

## Project Structure

```
genetic_algorithm/
│
├── main.py                 # Main execution script
├── ga/                    # Genetic Algorithm implementation
│   ├── __init__.py
│   └── genetic_algorithm.py
├── visualization/         # Visualization components
│   ├── __init__.py
│   └── visualizer.py
├── data_handler/         # Optimization functions
│   ├── __init__.py
│   └── functions.py
├── config/               # Configuration files
│   └── parameters.yaml
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/..../ga_project_template.git
cd genetic_algorithm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `config/parameters.yaml` file allows you to customize:

- Genetic algorithm parameters (population size, generations, etc.)
- Operator selection (crossover, mutation, selection methods)
- Optimization function choice
- Visualization settings

## Usage

Run the algorithm:
```bash
python main.py
```

The program will:
1. Load configuration from the YAML file
2. Display problem and algorithm information
3. Show real-time optimization progress
4. Generate a visualization of the fitness evolution
5. Save results and plots

## Features

- Modular genetic algorithm implementation
- Multiple selection methods (Tournament, Roulette, Rank)
- Various crossover operators (One-point, Two-point, Uniform)
- Different mutation types (Gaussian, Random Reset, Swap)
- Real-time fitness visualization
- Professional console output using Rich
- YAML-based configuration
- Three optimization test functions

## Dependencies

- NumPy: Numerical computations
- Matplotlib: Visualization
- Rich: Console interface
- PyYAML: Configuration management

## License

This project has been created by Nardone Emanuele