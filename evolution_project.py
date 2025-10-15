import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import to_rgb

# =========================
# Configurable parameters ^w^
# =========================
POPULATION_SIZE = 100         # number of AI "brains" per generation
GENERATIONS = 500             # total generations
ELITISM = 2                   # number of top brains that survive unchanged
MUTATION_RATE = 0.3           # mutation chance
TOURNAMENT_SIZE = 5           # selection pool size
HIDDEN_NEURONS = 6            # hidden layer size
ENV_TESTS = 5                 # how many random env tests per fitness evaluation
TRAIL = 50                    # default last N generations to show when Trail mode is enabled

#random seed for reproducibility, remove/comment out for different results each run
# random.seed(1)
# np.random.seed(1)

# =========================
# Environment definitions
# =========================
ENVIRONMENTS = [
    {"name": "Daytime heatwave", "inputs": [1.0, 0.0, 0.0], "target": [0.4, 0.7, 0.2]},
    {"name": "Cool night",       "inputs": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    {"name": "Dry drought",      "inputs": [1.0, 1.0, 0.0], "target": [0.8, 0.7, 0.0]},
    # trait outputs: [water retention, energy effiency, reproduction]
]

MIN_DURATION = 50
MAX_DURATION = 100

env_schedule = []
gen_count = 0
while gen_count < GENERATIONS:
    for base_env in ENVIRONMENTS:
        duration = random.randint(MIN_DURATION, MAX_DURATION)
        env_schedule.append((base_env, duration))
        gen_count += duration
        if gen_count >= GENERATIONS:
            break

# for plotting background bands
env_spans = []
cursor = 0
for base_env, duration in env_schedule:
    start = cursor
    end = min(cursor + duration, GENERATIONS)
    env_spans.append((start, end, base_env))
    cursor += duration

# visual helpers
env_colors = {
    "Daytime heatwave": "#FFDD99",
    "Cool night": "#99CCFF",
    "Dry drought": "#FF9999",
}
env_emojis = {"Daytime heatwave": "☀️", "Cool night": "🌙", "Dry drought": "💧"}

# =========================
# Neural network class
# =========================
class Brain:
    def __init__(self, output_neurons=3):
        self.w1 = np.random.randn(3, HIDDEN_NEURONS)
        self.w2 = np.random.randn(HIDDEN_NEURONS, output_neurons)
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
def fitness(brain):
    score = 0
    penalty = 0
    for _ in range(ENV_TESTS):
        g = np.random.uniform(0, GENERATIONS - 1)
        e = env_at_generation(g)
        varied_inputs = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in e["inputs"]]
        output, _ = brain.forward(varied_inputs)

        # Base accuracy component
        diff = [abs(o - t) for o, t in zip(output, e["target"])]
        score += 1 - np.mean(diff)

        # Punish for strong outputs where the target is near 0
        for o, t in zip(output, e["target"]):
            if t < 0.05 and o > 0.3:
                penalty += (o - t) * 0.5

    return max(0.0, (score / ENV_TESTS) - penalty)


def tournament_select(population, fitnesses):
    best = None; best_fit = -1
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randrange(len(population)); f = fitnesses[idx]
        if f > best_fit:
            best_fit = f; best = population[idx]
    return best

# =========================
# Smooth environment blending (cosine)
# =========================
def cosine_blend(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

def env_at_generation(gen):
    for i, (start, end, base_env) in enumerate(env_spans):
        if start <= gen < end:
            span_len = max(1, end - start)
            local_t = (gen - start) / span_len
            next_env = env_spans[(i + 1) % len(env_spans)][2]
            w = cosine_blend(local_t)
            blend_inputs = [(1 - w) * base_env["inputs"][k] + w * next_env["inputs"][k] for k in range(3)]
            blend_targets = [(1 - w) * base_env["target"][k] + w * next_env["target"][k] for k in range(len(base_env["target"]))]
            return {"name": base_env["name"], "inputs": blend_inputs, "target": blend_targets}
    return ENVIRONMENTS[0]

# =========================
# Initialize & evolve
# =========================
population = [Brain(len(ENVIRONMENTS[0]["target"])) for _ in range(POPULATION_SIZE)]

avg_fits, max_fits = [], []
best_outputs_history = {env['name']: [[] for _ in range(len(env['target']))] for env in ENVIRONMENTS}
best_hidden_history = {env['name']: [] for env in ENVIRONMENTS}
best_w1_history, best_w2_history = [], []

# Top K brains history
TOP_K = 5
brains_outputs_history = [
    {env['name']: [[] for _ in range(len(env['target']))] for env in ENVIRONMENTS}
    for _ in range(TOP_K)
]

env_names = [env['name'] for env in ENVIRONMENTS]
trait_names = ["Water", "Energy", "Reproduction"]

for gen in range(GENERATIONS):
    fits = [fitness(b) for b in population]
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

    # Record top K brains outputs for this generation
    for k in range(TOP_K):
        b = population[sorted_idx[k]]
        for env_obj in ENVIRONMENTS:
            out_k, _ = b.forward(env_obj['inputs'])
            for dim, val in enumerate(out_k):
                brains_outputs_history[k][env_obj['name']][dim].append(val)

    new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]
    while len(new_pop) < POPULATION_SIZE:
        parent = tournament_select(population, fits).copy()
        parent.mutate()
        new_pop.append(parent)
    population = new_pop

# =========================
# Visualization
# =========================
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(20,10))
gs = GridSpec(3, 2, figure=fig, width_ratios=[3.0, 1.2], height_ratios=[1,1,1])
plt.subplots_adjust(bottom=0.25, left=0.18, right=0.96, top=0.92)

ax_out = fig.add_subplot(gs[:, 0])
ax_hid = fig.add_subplot(gs[0, 1])
ax_w1  = fig.add_subplot(gs[1, 1])
ax_w2  = fig.add_subplot(gs[2, 1])

lines, target_lines, base_colors = {}, {}, {}

# Vivid base hues per brain
brain_base_colors = [
    np.array([0.90, 0.15, 0.15]),
    np.array([0.10, 0.35, 0.95]),
    np.array([0.10, 0.70, 0.20]),
    np.array([0.95, 0.55, 0.10]),
    np.array([0.60, 0.20, 0.85]),
]

# Shade factors per trait
shade_factors = [1.0, 0.8, 0.6]

# Create lines for top K brains
for k in range(TOP_K):
    for env_index, env_name in enumerate(env_names):
        for dim, trait in enumerate(trait_names):
            base = brain_base_colors[k]
            shade = np.clip(base * shade_factors[dim], 0, 1)
            line, = ax_out.plot([], [], '-', linewidth=(2.2 if k == 0 else 1.6),
                                color=shade, alpha=(1.0 if k == 0 else 0.65),
                                label=f"Brain {k+1} {env_name} {trait}", zorder=(3 if k==0 else 2))
            lines[(k, env_name, dim)] = line

# Target lines stay as reference
for env_index, env_name in enumerate(env_names):
    for dim, trait in enumerate(trait_names):
        tline, = ax_out.plot(range(GENERATIONS), [ENVIRONMENTS[env_index]["target"][dim]]*GENERATIONS,
                             '--', color=np.array(plt.get_cmap("tab10")(env_index * 3 + dim))[:3],
                             linewidth=1.0, alpha=0.6, zorder=1,
                             label=f"{env_name} {trait} target")
        target_lines[(env_name, dim)] = tline

ax_out.set_xlim(0, GENERATIONS)
ax_out.set_ylim(0, 1)
ax_out.set_xlabel("Generation")
ax_out.set_ylabel("Trait Output")
ax_out.set_title("Top 5 Brain Outputs")

# --- clean gradient background (no lines, no layering) ---
from matplotlib.colors import to_rgb

# create gradient image
width = GENERATIONS
height = 1  # just a single row
bg = np.zeros((height, width, 3))

def lerp(c1, c2, w):
    return (1 - w) * np.array(to_rgb(c1)) + w * np.array(to_rgb(c2))

# build color map smoothly along generations
for gen in range(GENERATIONS):
    e = env_at_generation(gen)
    bg[0, gen] = to_rgb(env_colors[e["name"]])

# blur the transitions slightly for smoothness
from scipy.ndimage import gaussian_filter1d
for ch in range(3):
    bg[0, :, ch] = gaussian_filter1d(bg[0, :, ch], sigma=3)

# draw as background
ax_out.imshow(bg, extent=[0, GENERATIONS, 0, 1],
              aspect='auto', alpha=0.25, origin='lower', zorder=0)

# clean labels and green dividers
for (start, end, base_env) in env_spans:
    ax_out.axvline(start, color=(0, 0.7, 0, 0.6), linewidth=1.1, zorder=9)  # bright semi-transparent green
    mid = (start + end) / 2
    ax_out.text(mid, -0.09, base_env["name"],
                transform=ax_out.get_xaxis_transform(),
                ha='center', va='top', fontsize=9,
                color='dimgray', fontweight='bold')

# --- Compact legend ---
legend_handles = []
for k in range(TOP_K):
    line_proxy, = ax_out.plot([], [], color=brain_base_colors[k],
                              linewidth=2.0, alpha=1.0,
                              label=f"Brain {k+1}")
    legend_handles.append(line_proxy)
for env_index, env_name in enumerate(env_names):
    for dim, trait in enumerate(trait_names):
        target_val = ENVIRONMENTS[env_index]["target"][dim]
        if target_val > 0.05:
            tcolor = np.array(to_rgb(env_colors[env_name]))
            trait_shade = np.clip(tcolor * (1.0 - 0.2 * dim), 0, 1)
            tproxy, = ax_out.plot([], [], '--', color=trait_shade, linewidth=1.5,
                                  alpha=0.8, label=f"{env_name} {trait} target")
            legend_handles.append(tproxy)
legend = ax_out.legend(handles=legend_handles, fontsize=9,
                       loc='upper right', framealpha=0.9)

# --- Generation number label ---
gen_text = ax_out.text(0.98, 1.03, '', transform=ax_out.transAxes,
                       ha='right', va='bottom', fontsize=12, fontweight='bold')

# --- Hidden activations ---
im_hid = ax_hid.imshow(np.zeros((HIDDEN_NEURONS, len(ENVIRONMENTS))), cmap='plasma', aspect='auto')
ax_hid.set_xticks(range(len(env_names))); ax_hid.set_xticklabels(env_names)
ax_hid.set_yticks(range(HIDDEN_NEURONS)); ax_hid.set_yticklabels([f'H{i}' for i in range(HIDDEN_NEURONS)])
ax_hid.set_title("Hidden Neuron Activations")
im_hid.set_clim(-1, 1)
fig.colorbar(im_hid, ax=ax_hid, fraction=0.046, pad=0.04)

# --- Weight heatmaps ---
im_w1 = ax_w1.imshow(np.zeros((3,HIDDEN_NEURONS)), vmin=-3, vmax=3, cmap="coolwarm")
ax_w1.set_title("Input→Hidden Weights"); fig.colorbar(im_w1, ax=ax_w1, fraction=0.046, pad=0.04)
im_w2 = ax_w2.imshow(np.zeros((HIDDEN_NEURONS,len(ENVIRONMENTS[0]["target"]))), vmin=-3, vmax=3, cmap="coolwarm")
ax_w2.set_title("Hidden→Output Weights"); fig.colorbar(im_w2, ax=ax_w2, fraction=0.046, pad=0.04)

# --- Style controls ---
style_state = {"thin": False, "markers": False, "trail": False, "filter_env": False}
def apply_style_to_lines():
    for key, line in lines.items():
        k = key[0]
        if style_state["thin"]:
            line.set_linewidth(1.3 if k > 0 else 1.7); line.set_alpha(0.6 if k > 0 else 0.9)
        else:
            line.set_linewidth(2.2 if k == 0 else 1.6); line.set_alpha(1.0 if k == 0 else 0.65)
        line.set_marker('o' if style_state["markers"] else '')
        line.set_markersize(4 if style_state["markers"] else 0)
    for tline in target_lines.values():
        tline.set_alpha(0.5 if style_state["thin"] else 0.7)

def apply_environment_filter(current_env_name):
    for (k, env_name, dim), line in lines.items():
        if style_state["filter_env"]:
            line.set_visible(env_name == current_env_name and brain_visibility[k] and trait_visibility[k][dim])
        else:
            line.set_visible(brain_visibility[k] and trait_visibility[k][dim])

def compute_x_range_for_gen(gen):
    return (max(0, gen - TRAIL) if style_state["trail"] else 0, gen)

def move_legend_dynamically(start_idx, end_idx):
    if style_state["trail"]:
        if (start_idx + end_idx) / 2 < GENERATIONS / 2:
            legend.set_loc('upper right')
        else:
            legend.set_loc('upper left')
    else:
        legend.set_loc('upper right')

# --- new trait toggles setup ---
rax_style = plt.axes([0.02, 0.63, 0.10, 0.26])
style_labels = ["Thin/Transparent", "Markers", "Trail mode", "Only Current Env"]
style_vis = [style_state["thin"], style_state["markers"], style_state["trail"], style_state["filter_env"]]
style_check = CheckButtons(rax_style, style_labels, style_vis)
rax_style.set_title("Style")

def style_toggle(label):
    if label == "Thin/Transparent":
        style_state["thin"] = not style_state["thin"]
    elif label == "Markers":
        style_state["markers"] = not style_state["markers"]
    elif label == "Trail mode":
        style_state["trail"] = not style_state["trail"]
    elif label == "Only Current Env":
        style_state["filter_env"] = not style_state["filter_env"]
    update(slider.val)
style_check.on_clicked(style_toggle)

rax_brains = plt.axes([0.02, 0.08, 0.10, 0.36])
brain_labels = [f"Brain {i}" for i in range(1, TOP_K+1)]
brain_visibility = [True, False, False, False, False]
brain_check = CheckButtons(rax_brains, brain_labels, brain_visibility)
rax_brains.set_title("Brains")

# make nested boxes for traits ^_^
trait_checks = []
trait_visibility = [[True, True, True] for _ in range(TOP_K)]
trait_boxes = []
for i in range(TOP_K):
    ax_trait = plt.axes([0.12, 0.36 - (0.07*i), 0.08, 0.05])
    c = CheckButtons(ax_trait, trait_names, trait_visibility[i])
    ax_trait.set_visible(False)
    trait_checks.append(c)
    trait_boxes.append(ax_trait)

def set_brain_visibility(idx, visible):
    for env_name in env_names:
        for dim in range(len(trait_names)):
            lines[(idx, env_name, dim)].set_visible(visible and trait_visibility[idx][dim])
    trait_boxes[idx].set_visible(visible)
    fig.canvas.draw_idle()

def toggle_trait(label, brain_idx):
    dim = trait_names.index(label)
    trait_visibility[brain_idx][dim] = not trait_visibility[brain_idx][dim]
    for env_name in env_names:
        lines[(brain_idx, env_name, dim)].set_visible(trait_visibility[brain_idx][dim])
    fig.canvas.draw_idle()

def brains_toggle(label):
    idx = int(label.split()[-1]) - 1
    brain_visibility[idx] = not brain_visibility[idx]
    set_brain_visibility(idx, brain_visibility[idx])

for i, c in enumerate(trait_checks):
    for l in c.labels:
        l.set_fontsize(8.5)
    c.on_clicked(lambda label, i=i: toggle_trait(label, i))

brain_check.on_clicked(brains_toggle)

# --- Slider and update ---
def update(val):
    gen = int(slider.val)
    gen_text.set_text(f"Generation: {gen}")
    env_obj = env_at_generation(gen)
    apply_environment_filter(env_obj["name"])
    start_idx, end_idx = compute_x_range_for_gen(gen)
    for k in range(TOP_K):
        for env_name in env_names:
            for dim in range(len(trait_names)):
                ydata = brains_outputs_history[k][env_name][dim][start_idx:end_idx+1]
                xdata = list(range(start_idx, end_idx+1))
                lines[(k, env_name, dim)].set_data(xdata, ydata)
    hid_data = np.array([best_hidden_history[env][gen] for env in env_names]).T
    im_hid.set_data(hid_data)
    im_w1.set_data(best_w1_history[gen])
    im_w2.set_data(best_w2_history[gen])
    apply_style_to_lines()
    move_legend_dynamically(start_idx, end_idx)
    fig.canvas.draw_idle()

ax_slider = plt.axes([0.22, 0.08, 0.58, 0.04])
slider = Slider(ax_slider, 'Generation', 0, GENERATIONS-1, valinit=GENERATIONS-1, valstep=1)
slider.on_changed(update)
ax_reset = plt.axes([0.81, 0.08, 0.08, 0.04])
Button(ax_reset, 'Reset').on_clicked(lambda e: slider.reset())

for i, vis in enumerate(brain_visibility):
    set_brain_visibility(i, vis)

update(GENERATIONS-1)

# --- subtle dotted grid overlay ---
for gx in np.arange(0, GENERATIONS + 1, 10):
    ax_out.plot([gx, gx], [0, 1],
                color=(0, 0, 0, 0.45),
                linewidth=0.6,
                linestyle=(0, (1.5, 3.5)),
                zorder=10)

for gy in np.arange(0, 1.05, 0.1):
    ax_out.plot([0, GENERATIONS], [gy, gy],
                color=(0, 0, 0, 0.45),
                linewidth=0.6,
                linestyle=(0, (1.5, 3.5)),
                zorder=10)

plt.show()
