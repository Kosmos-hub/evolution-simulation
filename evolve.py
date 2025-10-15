import numpy as np
import random
from utils import cosine_blend, env_at_generation
from environments import ENVIRONMENTS

# =========================
# Configurable parameters ^w^
# =========================
POPULATION_SIZE = 416
GENERATIONS = 1000
ELITISM = 10
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 30
HIDDEN_NEURONS = 6
ENV_TESTS = 15
TRAIL = 83

# =========================
# Neural network class
# =========================
class Brain:
    def __init__(self, output_neurons=3):
        self.w1 = np.random.uniform(-0.2, 0.2, (3, HIDDEN_NEURONS))
        self.w2 = np.random.uniform(-0.2, 0.2, (HIDDEN_NEURONS, output_neurons))



    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.w1))
        o = 1 / (1 + np.exp(-np.dot(h, self.w2)))
        return o, h

    def mutate(self, rate=MUTATION_RATE):
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate

    def copy(self):
        new = Brain(self.w2.shape[1])
        new.w1 = np.copy(self.w1)
        new.w2 = np.copy(self.w2)
        return new

# =========================
# Fitness
# =========================
def fitness(brain, env):
    varied_inputs = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in env["inputs"]]
    output, _ = brain.forward(varied_inputs)

    # use absolute error, not squared — then compress heavily
    diff = np.abs(np.array(output) - np.array(env["target"]))
    base = 1 - np.mean(diff)

    # make early scores very small, then accelerate improvement
    score = max(0.0, base ** 4)

    # penalize false positives on traits that should be off
    penalty = sum((o - t) * 0.8 for o, t in zip(output, env["target"]) if t < 0.05 and o > 0.25)
    return max(0.0, score - penalty)



# =========================
# Tournament selection
# =========================
def tournament_select(population, fitnesses):
    best = None
    best_fit = -1
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randrange(len(population))
        f = fitnesses[idx]
        if f > best_fit:
            best_fit = f
            best = population[idx]
    return best

# =========================
# Crossover
# =========================
def crossover(parent1, parent2):
    child = parent1.copy()
    mask_w1 = np.random.rand(*parent1.w1.shape) < 0.5
    mask_w2 = np.random.rand(*parent1.w2.shape) < 0.5
    child.w1 = np.where(mask_w1, parent1.w1, parent2.w1)
    child.w2 = np.where(mask_w2, parent1.w2, parent2.w2)
    return child
