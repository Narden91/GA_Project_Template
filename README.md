# Genetic Algorithm and Feature Selection Optimization Framework

This project implements a flexible and modular genetic algorithm framework for solving complex optimization problems. It features multiple encoding schemes, real-time visualization, and configurable parameters through a YAML configuration. Additionally, it includes a specialized Genetic Algorithm for feature selection, enabling efficient selection of relevant features for machine learning models.

## Overview

The framework provides a comprehensive genetic algorithm implementation with support for multiple encoding strategies (float, binary, and Gray binary), customizable operators, and real-time fitness visualization. It includes several challenging optimization functions like Ackley, Rosenbrock, and Schwefel functions. Moreover, it integrates a Feature Selection Genetic Algorithm (FeatureSelectionGA) that selects the most relevant features for a classification model, improving accuracy and reducing complexity. All parameters are configurable through a YAML file, making it simple to experiment with different settings and problems.

## Project Structure

```
genetic_algorithm/
│
├── main.py                 # Main execution script
├── ga/                     # Genetic Algorithm implementation
│   ├── __init__.py
│   ├── genetic_algorithm.py
│   ├── feature_selection_ga.py  # Genetic Algorithm for Feature Selection
│   ├── evolution_logger.py      # Logs evolution data
│   └── encoding.py        # Encoding schemes implementation
├── visualization/          # Visualization components
│   ├── __init__.py
│   └── visualizer.py
├── data_handler/           # Optimization functions & data loading
│   ├── __init__.py
│   ├── functions.py        # Test optimization functions
│   └── data_loader.py      # Data loading for feature selection
├── config/                 # Configuration files
│   └── parameters.yaml
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/.../ga_project_template.git
cd ga_project_template
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `config/parameters.yaml` file allows comprehensive customization of:

- Genetic algorithm parameters (population size, generations, etc.)
- Encoding settings (float, binary, or Gray binary encoding)
- Operator selection (crossover, mutation, selection methods)
- Optimization function choice
- Feature selection settings (enabling/disabling, dataset path)
- Visualization settings

### Encoding Options

The framework supports three encoding schemes:

1. **Float Encoding**: Direct representation of real numbers, suitable for continuous optimization problems. Parameters include precision settings.

2. **Binary Encoding**: Binary string representation of values, offering different genetic operator possibilities. Parameters include bits per variable.

3. **Gray Binary Encoding**: Uses Gray code to ensure adjacent values differ by only one bit, potentially improving optimization performance.

## Usage

### Running Standard Genetic Algorithm for Optimization
```bash
python main.py
```
The program will:
1. Load configuration from the YAML file
2. Display problem and algorithm information
3. Show real-time optimization progress with the chosen encoding
4. Generate a visualization of the fitness evolution
5. Save results and plots

### Running Feature Selection Genetic Algorithm

To enable feature selection, set `use_feature_selection: true` in `parameters.yaml`. The framework will:
1. Load the dataset specified in the configuration.
2. Initialize a Genetic Algorithm where each chromosome represents a feature subset.
3. Optimize the selection of relevant features to maximize classification accuracy.
4. Train a model (default: Decision Tree) on the best-selected feature subset and report accuracy.

## Features

### Core Components

The genetic algorithm includes:
- Modular implementation with pluggable components
- Multiple encoding schemes for different problem types
- Flexible operator selection
- Real-time fitness visualization
- YAML-based configuration
- Feature selection for machine learning models

### Genetic Operators

**Selection Methods:**
- Tournament Selection
- Roulette Wheel Selection
- Rank-based Selection

**Crossover Operators:**
- Single-point Crossover
- Two-point Crossover
- Uniform Crossover

**Mutation Operators:**
- Gaussian Mutation (Float encoding)
- Random Reset Mutation
- Swap Mutation
- Bit Flip Mutation (Binary encodings)

### Test Functions

The framework includes three challenging optimization functions:
- **Ackley Function**: Multi-modal test function with many local minima.
- **Rosenbrock Function**: Valley-shaped function with a global minimum inside a narrow parabolic valley.
- **Schwefel Function**: Deceptive function where the global minimum is far from the next best local minima.

### Feature Selection GA

- Uses a Genetic Algorithm where each gene represents the inclusion/exclusion of a feature.
- The fitness function optimizes classification accuracy while penalizing larger feature sets.
- Default classifier: Decision Tree (can be modified in `feature_selection_ga.py`).
- Logs feature selection evolution and results.
- Supports visualization of fitness evolution.

## Dependencies

- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Rich**: Console interface
- **PyYAML**: Configuration management
- **Scikit-learn**: Machine learning models for feature selection evaluation

## License

This project was created by Nardone Emanuele.

