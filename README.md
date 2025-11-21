# Topological Qubit Simulation

This project simulates topological qubits using Josephson junctions and minimal Kitaev chains. The repository contains implementations for both coupled Kitaev chain systems and minimal two-site Kitaev chains.

## Project Structure

### `josephson_junction/`
Simulates a topological qubit using a Josephson junction connecting two minimal Kitaev chains.

- **`analysis_andreev.ipynb`** - Jupyter notebook for running simulations and visualizing results
- **`functions.py`** - Helper functions used by the simulation code
- **`plotting.py`** - Plotting functions for various physical observables
- **`simulation_andreev.py`** - Core simulation code for the Josephson junction system
- **`sweep.py`** - Parameter sweep utilities
- **`figures/`** - Output directory for generated plots
- **`results/`** - Output directory for simulation data
- **`__pycache__/`** - Python cache files

### `minimal_Kitaev_chain/`
Simulates a singular two-site Kitaev chain.

- **`analysis.ipynb`** - Jupyter notebook for analysis and visualization
- **`calculator.py`** - Calculation utilities for the minimal chain
- **`functions.py`** - Helper functions specific to the minimal chain
- **`plotting.py`** - Plotting functions for observables
- **`simulation.py`** - Core simulation code for the two-site Kitaev chain
- **`figures/`** - Output directory for generated plots
- **`results/`** - Output directory for simulation data

### Usage

1. **For Josephson Junction simulations:**
   - Open `josephson_junction/analysis_andreev.ipynb`
   - Run the cells to execute simulations and generate plots

2. **For minimal Kitaev chain simulations:**
   - Open `minimal_Kitaev_chain/analysis.ipynb`
   - Run the cells to execute simulations and generate plots

### Workflow

Each module follows a similar structure:
1. `functions.py` - Contains specialized functions for physical calculations
2. `simulation*.py` - Main simulation logic
3. `plotting.py` - Visualization of observables
4. `analysis*.ipynb` - Interactive notebooks to run simulations and analyze results

## Contact

tornquistoscar@gmail.com