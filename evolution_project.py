import numpy as np
import random
from creature_visuals import show_env_fitness_map
from environments import ENVIRONMENTS
import matplotlib.pyplot as plt
from evolve import *
from visualize import show_visualization
from creature_visuals import show_dominance_map

if __name__ == "__main__":
    # =========================
    # Initialize & evolve
    # =========================
    population = [Brain(len(ENVIRONMENTS[0]["target"])) for _ in range(POPULATION_SIZE)]

    avg_fits, max_fits = [], []
    best_outputs_history = {env['name']: [[] for _ in range(len(env['target']))] for env in ENVIRONMENTS}
    best_hidden_history = {env['name']: [] for env in ENVIRONMENTS}
    best_w1_history, best_w2_history = [], []

    TOP_K = 5
    brains_outputs_history = [
        {env['name']: [[] for _ in range(len(env['target']))] for env in ENVIRONMENTS}
        for _ in range(TOP_K)
    ]

    env_names = [env['name'] for env in ENVIRONMENTS]
    trait_names = ["Water", "Energy", "Reproduction"]

    # =========================
    # Evolution loop :3
    # =========================
    for gen in range(GENERATIONS):
        current_env = env_at_generation(gen)
        fits = [fitness(b, current_env) for b in population]
        avg_fits.append(np.mean(fits))
        max_fits.append(np.max(fits))

        sorted_idx = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
        best_brain = population[sorted_idx[0]]

        for env_obj in ENVIRONMENTS:
            out, hidden = best_brain.forward(env_obj['inputs'])
            for dim, val in enumerate(out):
                best_outputs_history[env_obj['name']][dim].append(val)
            best_hidden_history[env_obj['name']].append(hidden.copy())

        best_w1_history.append(best_brain.w1.copy())
        best_w2_history.append(best_brain.w2.copy())

        for k in range(TOP_K):
            b = population[sorted_idx[k]]
            for env_obj in ENVIRONMENTS:
                out_k, _ = b.forward(env_obj['inputs'])
                for dim, val in enumerate(out_k):
                    brains_outputs_history[k][env_obj['name']][dim].append(val)

        new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            child = crossover(p1, p2)
            child.mutate()
            new_pop.append(child)

        population = new_pop

    # =========================
    # Post-processing
    # =========================
    stability = []
    for gen in range(GENERATIONS):
        vals = []
        for env in ENVIRONMENTS:
            for dim_data in best_outputs_history[env["name"]]:
                vals.append(dim_data[gen])
        stability.append(np.std(vals))

    corr_history = []
    for gen in range(GENERATIONS):
        data = []
        for env in ENVIRONMENTS:
            data.append([best_outputs_history[env["name"]][i][gen] for i in range(len(trait_names))])
        data = np.array(data)
        corr = np.corrcoef(data.T)
        corr_history.append(corr)

    # =========================
    # Visualization
    # =========================
    main_fig = show_visualization(
        avg_fits, max_fits, stability, corr_history,
        best_hidden_history, brains_outputs_history,
        env_names, trait_names, TOP_K, GENERATIONS, HIDDEN_NEURONS
    )

    milestones = [int(GENERATIONS * x) - 1 for x in [0.25, 0.5, 0.75, 1.0]]

    env_map_fig = show_env_fitness_map(brains_outputs_history, ENVIRONMENTS, fitness, milestones, TOP_K)
    dom_map_fig = show_dominance_map(brains_outputs_history, ENVIRONMENTS, TOP_K, GENERATIONS)

    # show all at once (non-blocking handled inside)
    plt.show()


