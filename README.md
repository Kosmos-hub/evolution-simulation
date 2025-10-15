# Evolutionary Neural Network Simulation :3

lil guy simulates evolution — neural “brains” learn to survive under changing environments.  
They mutate, compete, and sometimes recombine through crossover to adapt over time.

##  What It Does
- Evolving a whole population of neural networks over hundreds of generations.
- Each generation lives in a certain environment (*Daytime Heatwave*, *Cool Night*, *Dry Drought*).
- Fitness is based only on how well each brain fits the **current** environment.
- Includes genetic crossover, mutations, and elitism.
- Interactive Matplotlib dashboard that shows:
  - Each brain’s trait outputs over time.
  - Hidden neuron activations.
  - Fitness history and trait stability.
  - Correlation between traits (with environment bands).
  - Checkboxes to toggle brains & traits live.
  - A slider to move through generations.
  - Snapshot button (for pretty evolution pics).

##  Reading the Graph
It’s a bit busy at first glance 3';
The big graph in the center shows traits of the top 5 brains over time.  
Each brain has 3 traits — **Water**, **Energy**, and **Reproduction** — and each environment has different target levels for those traits.

The background hue changes with the environment, so you can see the population adapting as conditions shift.  
Use the checkboxes to hide stuff or focus on a single brain or trait.

##  Environments
# target trait outputs: [water retention, energy effiency, reproduction]
1.  **Daytime Heatwave** (target of [0.4, 0.7, 0.2])
2.  **Cool Night**       (target of [0.0, 0.0, 1.0])
3.  **Dry Drought**      (target of [0.8, 0.7, 0.0])

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- SciPy (optional, for smoother transitions)

```bash
pip install numpy matplotlib scipy

Notes

Run python main.py to start the simulation.
The program will evolve, then show the interactive plot automatically.

You can drag the generation slider to explore or hit “Snapshot” to save an image of that moment in evolution.