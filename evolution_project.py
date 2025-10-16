import numpy as np
import json
import gzip
import shutil
from datetime import datetime
import random

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from environments import ENVIRONMENTS
from utils import env_at_generation
from evolve import *
from visualize import show_visualization
from creature_visuals import show_env_fitness_map, show_dominance_map

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

    # full population log
    all_brains_history = []

    # random immigrants each gen (~5% of pop)
    IMMIGRANTS_PER_GEN = max(1, POPULATION_SIZE // 20)

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

        # multi-env fitness
        fits = [fitness_multi(b, current_env) for b in population]

        # fitness sharing (niching) on phenotype (current outputs)
        phenos = np.array([b.forward(current_env["inputs"])[0] for b in population])  # (N, 3)
        D = cdist(phenos, phenos, metric="euclidean")
        sigma = 0.25
        sharing = 1.0 / (1.0 + (D / sigma) ** 2)
        density = sharing.sum(axis=1)
        fits_shared = np.array(fits) / (density + 1e-6)

        # record full population snapshot
        gen_record = {"generation": gen, "env": current_env["name"], "brains": []}
        for i, b in enumerate(population):
            out, hidden = b.forward(current_env["inputs"])
            gen_record["brains"].append({
                "id": i,
                "w1": b.w1.tolist(),
                "w2": b.w2.tolist(),
                "fitness": float(fits_shared[i]),
                "raw_fitness": float(fits[i]),
                "output": out.tolist(),
                "hidden": hidden.tolist(),
            })
        all_brains_history.append(gen_record)

        # metrics (use raw fitness for readability)
        avg_fits.append(float(np.mean(fits)))
        max_fits.append(float(np.max(fits)))

        # selection indices by shared fitness
        sorted_idx = list(np.argsort(-fits_shared))

        # capture the current best brain’s outputs across all envs
        best_brain = population[sorted_idx[0]]
        for env_obj in ENVIRONMENTS:
            out, hidden = best_brain.forward(env_obj['inputs'])
            for dim, val in enumerate(out):
                best_outputs_history[env_obj['name']][dim].append(float(val))
            best_hidden_history[env_obj['name']].append(hidden.copy())
        best_w1_history.append(best_brain.w1.copy())
        best_w2_history.append(best_brain.w2.copy())

        # track TOP_K brains output history
        for k in range(min(TOP_K, len(population))):
            b = population[sorted_idx[k]]
            for env_obj in ENVIRONMENTS:
                out_k, _ = b.forward(env_obj['inputs'])
                for dim, val in enumerate(out_k):
                    brains_outputs_history[k][env_obj['name']][dim].append(float(val))

        # build next population
        new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]

        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(population, fits_shared)
            p2 = tournament_select(population, fits_shared)
            child = crossover(p1, p2)
            new_pop.append(child)

        # adaptive mutation for non-elites
        rate = schedule_mutation_rate(gen, GENERATIONS)
        for child in new_pop[ELITISM:]:
            child.mutate(rate=rate)

        # random immigrants (replace non-elites)
        for _ in range(IMMIGRANTS_PER_GEN):
            idx = random.randrange(ELITISM, len(new_pop))
            new_pop[idx] = Brain(len(ENVIRONMENTS[0]["target"]))

        population = new_pop

    # =========================
    # Post-processing
    # =========================
    stability = []
    for g in range(GENERATIONS):
        vals = []
        for env in ENVIRONMENTS:
            for dim_data in best_outputs_history[env["name"]]:
                vals.append(dim_data[g])
        stability.append(float(np.std(vals)))

    corr_history = []
    for g in range(GENERATIONS):
        data = []
        for env in ENVIRONMENTS:
            data.append([best_outputs_history[env["name"]][i][g] for i in range(len(trait_names))])
        data = np.array(data)
        corr_history.append(np.corrcoef(data.T))

    results = {
        "avg_fits": avg_fits,
        "max_fits": max_fits,
        "stability": stability,
        "corr_history": [c.tolist() for c in corr_history],
        "best_hidden_history": {k: [np.array(x).tolist() for x in v] for k, v in best_hidden_history.items()},
        "brains_outputs_history": {
            str(k): {
                env: [list(trait_series) for trait_series in env_data]
                for env, env_data in entry.items()
            } for k, entry in enumerate(brains_outputs_history)
        },
        "parameters": {
            "POPULATION_SIZE": POPULATION_SIZE,
            "GENERATIONS": GENERATIONS,
            "HIDDEN_NEURONS": HIDDEN_NEURONS,
            "ELITISM": ELITISM,
            "MUTATION_RATE_START": MUTATION_RATE_START,
            "MUTATION_RATE_END": MUTATION_RATE_END,
            "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
            "ENV_TESTS": ENV_TESTS,
        },
        "timestamp": datetime.now().isoformat()
    }

    # =========================
    # Save experiment data
    # =========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_filename = f"all_brains_dump_{timestamp}.json"
    with open(raw_filename, "w") as f:
        json.dump(all_brains_history, f, indent=2)
    print(f"Saved full population history → {raw_filename}")

    gz_filename = raw_filename + ".gz"
    with open(raw_filename, "rb") as f_in, gzip.open(gz_filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Compressed dump → {gz_filename}")

    exp_filename = f"experiment_dump_{timestamp}.json"
    with open(exp_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment summary → {exp_filename}")

    # =========================
    # Visualization
    # =========================
    main_fig = show_visualization(
        avg_fits, max_fits, stability, corr_history,
        best_hidden_history, brains_outputs_history,
        env_names, trait_names, TOP_K, GENERATIONS, HIDDEN_NEURONS
    )

    milestones = [int(GENERATIONS * x) - 1 for x in [0.25, 0.5, 0.75, 1.0]]
    show_env_fitness_map(brains_outputs_history, ENVIRONMENTS, fitness, milestones, TOP_K)
    show_dominance_map(brains_outputs_history, ENVIRONMENTS, TOP_K, GENERATIONS)

    plt.show()