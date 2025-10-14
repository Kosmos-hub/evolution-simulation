import random
import math
import statistics
import matplotlib.pyplot as plt

# WeeWoo important these are the configureable parameters ^w^
POPULATION_SIZE = 200 
GENOME_LENGTH = 20 # number of genes, floats betwixt 0 and 1 ;3
GENERATIONS = 200
MUTATION_RATE = .05 # probability per gene for mutation
MUTATION_STRENGTH = .1 # gaussian *;D stddev for mutation amount
TOURNAMENT_SIZE = 5 # for selection
ELITISM = 2 # top number copied unchanged each generation
USE_CROSSOVER = True
CROSSOVER_RATE = .9

random.seed(1)

def random_genome():
    return [random.random() for _ in range(GENOME_LENGTH)]

# okay boom full fucking DESERT function baby lets do ts
ENVIRONMENTS = [
    {"name": "Daytime heatwave", "weights": [2.0, 0.1, 0.0]}, # heat matters most
    {"name": "Cool night",       "weights": [0.0, 0.06, 1.2]}, # night activity matters most
    {"name": "Dry drought",      "weights": [0.8, 2.0, 0.0]}, # water storage matters most
]

# random environment durations 
MIN_DURATION = 10 
MAX_DURATION = 30

env_schedule = []
gen_count = 0
while gen_count < GENERATIONS:
    for base_env in ENVIRONMENTS:
        duration = random.randint(MIN_DURATION, MAX_DURATION)
        env_schedule.append((base_env, duration))
        gen_count += duration
        if gen_count >= GENERATIONS:
            break

# adjust genome length if needed
GENOME_LENGTH = len(ENVIRONMENTS[0]["weights"])

def environment_for_generation(gen):
    # more complicated environments time woooo
    gen_cursor = 0
    for base_env, duration in env_schedule:
        if gen_cursor <= gen-1 < gen_cursor + duration:
            env = {"name": base_env["name"], "weights": []}
            # apply your existing ±20–30% variation
            for w in base_env["weights"]:
                variation = w * random.uniform(-0.2, 0.2)
                env["weights"].append(max(0.0, w + variation))
            return env
        gen_cursor += duration
    return ENVIRONMENTS[0]  # fallback


def fitness(genome, env):
    """Evaluate survival ability under current desert conditions."""
    weights = env["weights"]
    # weighted match to ideal "1.0" per trait
    score = sum(w * g for w, g in zip(weights, genome))
    # optional small penalty for being extreme (simulates getting bigger :3)
    penalty = .065 * sum(abs(g - .5) for g in genome)
    return score - penalty

# MUTATION TIMEEEEE
def mutate(genome):
    new = genome[:]
    for i in range(len(new)):
        if random.random() < MUTATION_RATE:
            new[i] += random.gauss(0, MUTATION_STRENGTH)
            # clamp to [0,1]
            new[i] = max(0.0, min(1.0, new[i]))
    return new

def crossover(a, b):
    if random.random() > CROSSOVER_RATE:
        return a[:]
    # single point crossover
    pt = random.randint(1, GENOME_LENGTH - 1)
    return a[:pt] + b[pt:]

def tournament_select(population, fitnesses):
    # pick k random and return best
    best = None
    best_fit = -1
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randrange(len(population))
        f = fitnesses[idx]
        if f > best_fit:
            best_fit = f
            best = population[idx]
    return best

# initializing population *^* 
population = [random_genome() for _ in range(POPULATION_SIZE)]

# Stat track
avg_fits = []
max_fits = []

avg_genes_history = []

for gen in range(1, GENERATIONS + 1):
    # evaluate
    env = environment_for_generation(gen)
    fits = [fitness(g, env) for g in population]
    avg = statistics.mean(fits)
    mx = max(fits)
    avg_fits.append(avg)
    max_fits.append(mx)
    avg_genes = [sum(g[i] for g in population) / POPULATION_SIZE for i in range(GENOME_LENGTH)]
    avg_genes_history.append(avg_genes)

    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}: env={env['name']:<15} avg={avg:.4f}, max={mx:.4f}")

    # create next generation
    # Elitism: keep top n genomes
    sorted_idx = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
    new_pop = [population[i][:] for i in sorted_idx[:ELITISM]]

    # fill the rest
    while len(new_pop) < POPULATION_SIZE:
        parent_a = tournament_select(population, fits)
        parent_b = tournament_select(population, fits)
        child = crossover(parent_a, parent_b) if USE_CROSSOVER else parent_a[:]
        child = mutate(child)
        new_pop.append(child)

    population = new_pop

# --- Trait names ---
trait_names = ["Heat Resistance", "Water Storage", "Night Activity"]

# after run, print best genome and show plot 
env = environment_for_generation(gen)
fits = [fitness(g, env) for g in population]
best_idx = max(range(len(population)), key=lambda i: fits[i])
best = population[best_idx]
print("\nBest fitness: ", fitness(best, env))
print("Best genome (first 10 genes):", best[:10])

# Plot fitness over generations
plt.figure(figsize=(8,4))
plt.plot(avg_fits, label='avg fitness')
plt.plot(max_fits, label='max fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolution progress ;3')
plt.tight_layout()
plt.show()

# plotting genes babyyy
# --- Environment colors ---
env_colors = {
    "Daytime heatwave": "#FFDD99",  # pale orange
    "Cool night": "#99CCFF",        # pale blue
    "Dry drought": "#FF9999",       # light red
}

plt.figure(figsize=(12,6))

# plot each gene line with its trait name
for i, trait in enumerate(trait_names):
    plt.plot([avg_genes_history[g][i] for g in range(GENERATIONS)], label=trait, linewidth=2)

# add colored environment background bands
gen_cursor = 0
for base_env, duration in env_schedule:
    plt.axvspan(gen_cursor, min(gen_cursor + duration, GENERATIONS), 
                color=env_colors[base_env['name']], alpha=0.15)
    gen_cursor += duration

# Add a legend for the environments
from matplotlib.patches import Patch
env_legend = [Patch(facecolor=color, alpha=0.3, label=name) for name, color in env_colors.items()]
plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + env_legend,
           bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)

plt.xlabel('Generation')
plt.ylabel('Average gene value')
plt.title('Desert Trait Evolution Over Generations')
plt.tight_layout()
plt.show()