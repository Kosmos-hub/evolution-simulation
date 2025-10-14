# Evolutionary Neural Network Simulation :3

This project implements a simple evolutionary neural network that learns to adapt to different environmental conditions over generations. The neural network (Brain) evolves through mutation and tournament selection to better match target outputs for a variety of simulated environments.

## Features XD

- Evolution of a population of neural networks over multiple generations.
- Dynamic fitness evaluation with random environmental variations.
- Interactive visualization with Matplotlib:
  - Brain outputs per environment.
  - Hidden neuron activations heatmap.
  - Input > Hidden and Hidden > Output weights heatmaps.
  - Slider to explore different generations.

## To read the graph
uh its kinda complicated, basically the big graph in the center has 
two boxes of checkboxes, one containing different filters to make the
graph more legible, and the other shows lines based off different genes.
if you want more info about them read my source code or ask me and id love to explain :3 but basically there are 3 genes (outputs) per brain and each environment requires different results of each of those outputs (simple stuff like [.2,.6,.9]) but we can think of these as importances of different traits depending on the environment theyre in (we can think of this like at night we want animals to reproduce so we make it [0.0,0.0,1.0] meaning the last trait (reproduction) is most important).
the graph shows the results of three brains: the best brain for night, the best brain for the heatwave, and the best brain for the drought. it then subdivides these brains into their outputs for each trait. this is then displayed onto a containing generation n on the x-axis and fitness (output) on the y-axis, with the graph being hued to represent the change in environments.

## Environments ;3

1. **Daytime heatwave** 
2. **Cool night** 
3. **Dry drought** 

## Requirements ;b

- Python 3.8+
- Numpy
- Matplotlib

Install dependencies via:

```bash
pip install numpy matplotlib
