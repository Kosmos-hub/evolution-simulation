# creature_visuals.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def show_dominance_map(brains_outputs_history, ENVIRONMENTS, TOP_K, GENERATIONS):
    n_envs = len(ENVIRONMENTS)
    dom_map = np.zeros((n_envs, GENERATIONS, 3))

    cmap = plt.get_cmap("tab10", TOP_K)

    for gen in range(GENERATIONS):
        for e_idx, env in enumerate(ENVIRONMENTS):
            scores = []
            for k in range(TOP_K):
                outs = [brains_outputs_history[k][env['name']][dim][gen]
                        for dim in range(len(env['target']))]
                diff = np.abs(np.array(outs) - np.array(env['target']))
                score = 1 - np.mean(diff)
                scores.append(score)
            scores = np.array(scores)

            winner = np.argmax(scores)
            second = np.partition(scores, -2)[-2]
            gap = np.clip((scores[winner] - second) * 4, 0, 1)

            winner_col = np.array(cmap(winner)[:3])
            neutral = np.array([0.5, 0.5, 0.5])
            blended = neutral * (1 - gap) + winner_col * gap
            dom_map[e_idx, gen, :] = blended

    fig, ax = plt.subplots(figsize=(12, 3 + 0.4 * n_envs))
    ax.imshow(dom_map, aspect="auto", origin="upper")

    ax.set_yticks(range(n_envs))
    ax.set_yticklabels([env["name"] for env in ENVIRONMENTS])
    ax.set_xlabel("Generation")
    ax.set_title("Environment Dominance Map (Blended)")

    handles = [plt.Line2D([0], [0], color=cmap(k), lw=6, label=f"Brain {k+1}") for k in range(TOP_K)]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Dominant Brain")

    plt.tight_layout()
    plt.show(block=False)
    return fig


def show_env_fitness_map(brains_outputs_history, ENVIRONMENTS, fitness_fn, generation_points, TOP_K):
    """
    Visualizes the fitness of top-K brains across all environments at milestone generations.
    """
    n_envs = len(ENVIRONMENTS)
    fig, axs = plt.subplots(1, len(generation_points), figsize=(16, 4))
    fig.suptitle("Environmental Fitness Map (Top-K Brains)", fontsize=14, weight='bold')

    for ax, gen in zip(axs, generation_points):
        heat = np.zeros((n_envs, TOP_K))
        for e_idx, env in enumerate(ENVIRONMENTS):
            for k in range(TOP_K):
                # approximate fitness by comparing brain outputs at that gen to env targets
                outs = [brains_outputs_history[k][env['name']][dim][gen] for dim in range(len(env['target']))]
                diff = np.abs(np.array(outs) - np.array(env['target']))
                score = 1 - np.mean(diff)
                heat[e_idx, k] = np.clip(score, 0, 1)

        im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for (i, j), val in np.ndenumerate(heat):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=8, weight='bold')
        ax.set_title(f"Generation {gen}")
        ax.set_xticks(range(TOP_K))
        ax.set_xticklabels([f"B{k+1}" for k in range(TOP_K)])
        ax.set_yticks(range(n_envs))
        ax.set_yticklabels([env["name"] for env in ENVIRONMENTS])

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label='Fitness')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show(block=False)
    return fig


def draw_creature(ax, water, energy, reproduction, idx):
    """Draw one creature on its axis based on trait values"""
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    # body color: water controls hue
    base_color = (1 - water, 0.3, water)  # dry→red, wet→blue
    size = 0.3 + 0.4 * energy             # energy → body radius

    body = Circle((0, 0), size, color=base_color, alpha=0.8)
    ax.add_patch(body)

    # reproduction satellites
    n = int(1 + reproduction * 6)
    for i in range(n):
        angle = 2 * np.pi * i / n
        r = size * 1.6
        sx, sy = r * np.cos(angle), r * np.sin(angle)
        sat = Circle((sx, sy), 0.08, color=base_color, alpha=0.6)
        ax.add_patch(sat)

    ax.text(0, -1.05, f"Creature {idx+1}", ha='center', va='top',
            fontsize=8, color='dimgray')


def show_topk_creatures(brains_outputs_history, env_names, trait_names, TOP_K, GENERATIONS):
    """Create figure with TOP_K creature subplots that update with the slider"""
    fig, axes = plt.subplots(1, TOP_K, figsize=(TOP_K*2.5, 3))
    if TOP_K == 1:
        axes = [axes]

    def update_creatures(gen):
        for k in range(TOP_K):
            # take average across environments for visible representation
            avg_traits = np.mean(
                [[brains_outputs_history[k][env][i][gen]
                  for i in range(len(trait_names))]
                 for env in env_names],
                axis=0)
            draw_creature(axes[k], *avg_traits, idx=k)
        fig.canvas.draw_idle()

    return fig, update_creatures

import matplotlib.pyplot as plt

def show_milestone_creatures(brains_outputs_history, env_names, trait_names, TOP_K, GENERATIONS):
    # create separate window for milestones
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Milestone Creatures (25%, 50%, 75%, 100%)", fontsize=14, fontweight='bold')
    
    gens = [
        int(GENERATIONS * 0.25),
        int(GENERATIONS * 0.50),
        int(GENERATIONS * 0.75),
        GENERATIONS - 1
    ]
    titles = ["25%", "50%", "75%", "100%"]

    for ax, gen, title in zip(axes.flat, gens, titles):
        ax.set_title(f"Generation {title}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

        for k in range(TOP_K):
            # gather trait outputs for this brain at this generation
            vals = [brains_outputs_history[k][env_names[0]][i][gen] for i in range(len(trait_names))]
            water, energy, repro = vals

            # base color blending similar to env tones
            base_col = np.array([water, energy, repro])
            col = base_col / (np.max(base_col) + 1e-6)
            color = (col[0], col[1], col[2], 0.8)

            # ellipse shape based on traits
            width = 0.25 + 0.6 * water
            height = 0.25 + 0.6 * energy
            angle = repro * 180  # rotation hint
            e = Ellipse((0, 0), width, height, angle=angle, color=color, lw=1.0, ec='black', alpha=0.9)

            # offset each creature around circle so they don’t overlap
            theta = 2 * np.pi * k / TOP_K
            e.set_center((np.cos(theta) * 0.7, np.sin(theta) * 0.7))

            ax.add_patch(e)

        ax.text(0, -1.05, f"Gen {gen}", ha='center', va='top', fontsize=9, color='gray')


    plt.tight_layout(rect=[0, 0, 0.95, 1])
    legend_elements = [
    Patch(facecolor='red', edgecolor='k', label='High Water output'),
    Patch(facecolor='green', edgecolor='k', label='High Energy output'),
    Patch(facecolor='blue', edgecolor='k', label='High Reproduction output'),
    Line2D([0], [0], marker='o', color='w', label='Larger = stronger overall fitness',
           markerfacecolor='gray', markersize=10)
]

    fig.legend(handles=legend_elements, loc='upper center',
           ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.98))
    return fig
