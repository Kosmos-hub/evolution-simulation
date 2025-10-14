# Evolutionary Neural Network Simulation 🌞🌙💧

This project implements a simple evolutionary neural network that learns to adapt to different environmental conditions over generations. The neural network (Brain) evolves through mutation and tournament selection to better match target outputs for a variety of simulated environments.

## Features

- Evolution of a population of neural networks over multiple generations.
- Dynamic fitness evaluation with random environmental variations.
- Interactive visualization with Matplotlib:
  - Brain outputs per environment.
  - Hidden neuron activations heatmap.
  - Input > Hidden and Hidden > Output weights heatmaps.
  - Slider to explore different generations.

## Environments

1. **Daytime heatwave** ☀️
2. **Cool night** 🌙
3. **Dry drought** 💧

## Requirements

- Python 3.8+
- Numpy
- Matplotlib

Install dependencies via:

```bash
pip install numpy matplotlib
