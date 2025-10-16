import numpy as np
import random
from random import choices
from utils import cosine_blend, env_at_generation
from environments import ENVIRONMENTS


# =========================
# Configurable parameters ^w^
# =========================
POPULATION_SIZE = 80
GENERATIONS = 200
ELITISM = 2

# Adaptive mutation schedule
MUTATION_RATE_START = 0.25
MUTATION_RATE_END   = 0.05

# Selection pressure (reduced slightly)
TOURNAMENT_SIZE = 4

HIDDEN_NEURONS = 6
ENV_TESTS = 5
TRAIL = 15

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

    def mutate(self, rate=MUTATION_RATE_END):
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate
        # L2 weight decay to avoid bloat
        decay = 0.995
        self.w1 *= decay
        self.w2 *= decay

    def copy(self):
        new = Brain(self.w2.shape[1])
        new.w1 = np.copy(self.w1)
        new.w2 = np.copy(self.w2)
        return new

# =========================
# Fitness (single env)
# =========================
def fitness(brain, env):
    # slight input jitter
    varied_inputs = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in env["inputs"]]
    output, _ = brain.forward(varied_inputs)

    diff = np.abs(np.array(output) - np.array(env["target"]))
    base = 1 - np.mean(diff)

    # compress early wins; encourage gradual improvement
    score = max(0.0, base ** 4)

    # penalize false positives on traits that should be off
    penalty = sum((o - t) * 0.8 for o, t in zip(output, env["target"]) if t < 0.05 and o > 0.25)
    return max(0.0, score - penalty)

# =========================
# Schedule mutation rate
# =========================
def schedule_mutation_rate(gen, total_gens):
    t = gen / max(1, total_gens - 1)
    return MUTATION_RATE_START * (1 - t) + MUTATION_RATE_END * t

# =========================
# Fitness across envs
# =========================
def fitness_multi(brain, current_env, k=ENV_TESTS):
    pool = [current_env] + choices(ENVIRONMENTS, k=max(0, k - 1))
    scores = [fitness(brain, e) for e in pool]
    return float(np.mean(scores))

# =========================
# Tournament selection
# =========================
def tournament_select(population, fitnesses):
    best = None
    best_fit = -1.0
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
