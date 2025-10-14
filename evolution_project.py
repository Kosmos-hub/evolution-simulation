import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import to_rgb


# =========================
# Configurable parameters ^w^
# =========================
POPULATION_SIZE = 100 # number of AI "brains" per generation
GENERATIONS = 200 # total generations
ELITISM = 2 # number of top brains that survive unchanged
MUTATION_RATE = 0.3 # mutation intensity
TOURNAMENT_SIZE = 5 # selection pool size
HIDDEN_NEURONS = 6 # hidden layer size
ENV_TESTS = 5 # how many random env tests per fitness evaluation
TRAIL = 20            # default last N generations to show when Trail mode is enabled

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
]

MIN_DURATION = 10
MAX_DURATION = 30

# generate environment schedule per generation
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
        # input>hidden, hidden>output weights
        self.w1 = np.random.randn(3, HIDDEN_NEURONS)
        self.w2 = np.random.randn(HIDDEN_NEURONS, output_neurons)
    def forward(self, inputs):
        """forward pass: returns output and hidden activations"""
        h = np.tanh(np.dot(inputs, self.w1))
        o = 1 / (1 + np.exp(-np.dot(h, self.w2)))
        return o, h
    def mutate(self, rate=MUTATION_RATE):
        """mutate weights slightly to keep evolution silly :3 teehee"""
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate
    def copy(self):
        """deep copy for elitism / reproduction"""
        new = Brain(self.w2.shape[1])
        new.w1 = np.copy(self.w1)
        new.w2 = np.copy(self.w2)
        return new

# =========================
# Fitness
# =========================
def fitness(brain):
    score = 0
    for _ in range(ENV_TESTS):
        # pick a random point in time for smooth environment blending
        g = np.random.uniform(0, GENERATIONS - 1)
        e = env_at_generation(g)
        # add small random noise to inputs
        varied_inputs = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in e["inputs"]]
        output, _ = brain.forward(varied_inputs)
        score += 1 - np.mean([abs(o - t) for o, t in zip(output, e["target"])])
    return score / ENV_TESTS

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
    """t in [0,1] → smoothstep-ish blend (0→1 with soft ends)."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

def env_at_generation(gen):
    """
    Return a blended environment dict for generation 'gen':
    inputs/targets are a cosine blend of the active span and the next span.
    """
    # find which span gen is in
    total = 0
    for i, (start, end, base_env) in enumerate(env_spans):
        if start <= gen < end:
            # blend factor in this span
            span_len = max(1, end - start)
            local_t = (gen - start) / span_len  # 0..1 in this span
            # blend toward the NEXT span (loops around)
            next_env = env_spans[(i + 1) % len(env_spans)][2]
            w = cosine_blend(local_t)
            # blend inputs/targets elementwise
            blend_inputs = [
                (1 - w) * base_env["inputs"][k] + w * next_env["inputs"][k] for k in range(3)
            ]
            blend_targets = [
                (1 - w) * base_env["target"][k] + w * next_env["target"][k] for k in range(len(base_env["target"]))
            ]
            return {"name": base_env["name"], "inputs": blend_inputs, "target": blend_targets, "w": w, "next_name": next_env["name"], "span": (start, end)}
    # fallback
    return {"name": ENVIRONMENTS[0]["name"], "inputs": ENVIRONMENTS[0]["inputs"], "target": ENVIRONMENTS[0]["target"], "w": 0.0, "next_name": ENVIRONMENTS[1]["name"], "span": (0, GENERATIONS)}


# =========================
# Initialize & evolve
# =========================
population = [Brain(len(ENVIRONMENTS[0]["target"])) for _ in range(POPULATION_SIZE)]

avg_fits = []; max_fits = []
best_outputs_history = {env['name']: [[] for _ in range(len(env['target']))] for env in ENVIRONMENTS}
best_hidden_history = {env['name']: [] for env in ENVIRONMENTS}
best_w1_history = []; best_w2_history = []

env_names = [env['name'] for env in ENVIRONMENTS]
trait_names = ["Water", "Energy", "Reproduction"]

# =========================
# Evolution loop
# =========================
for gen in range(GENERATIONS):
    # evaluate population with blended environments
    fits = [fitness(b) for b in population]
    avg_fits.append(np.mean(fits))
    max_fits.append(np.max(fits))
    best_idx = int(np.argmax(fits))
    best_brain = population[best_idx]

    # record best brain outputs on base environments
    for env_obj in ENVIRONMENTS:
        out, hidden = best_brain.forward(env_obj['inputs'])
        for dim, val in enumerate(out):
            best_outputs_history[env_obj['name']][dim].append(val)
        best_hidden_history[env_obj['name']].append(hidden.copy())

    # save weights for visualization
    best_w1_history.append(best_brain.w1.copy())
    best_w2_history.append(best_brain.w2.copy())

    # selection and mutation
    sorted_idx = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
    new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]
    while len(new_pop) < POPULATION_SIZE:
        parent = tournament_select(population, fits).copy()
        parent.mutate(rate=MUTATION_RATE)
        new_pop.append(parent)
    population = new_pop

# =========================
# Visualization
# =========================
from matplotlib.gridspec import GridSpec

# Layout: big main chart on the left, 3 stacked panels on the right
fig = plt.figure(figsize=(20,10))
gs = GridSpec(3, 2, figure=fig, width_ratios=[3.0, 1.2], height_ratios=[1,1,1])
plt.subplots_adjust(bottom=0.25, left=0.18, right=0.96, top=0.92)

ax_out = fig.add_subplot(gs[:, 0])       # spans all rows on left
ax_hid = fig.add_subplot(gs[0, 1])       # right column, row 1
ax_w1  = fig.add_subplot(gs[1, 1])       # right column, row 2
ax_w2  = fig.add_subplot(gs[2, 1])       # right column, row 3

# --- Output lines & targets ---
lines = {}
target_lines = {}
for env_index, env_name in enumerate(env_names):
    base_color = np.array(plt.get_cmap("tab10")(env_index))[:3]
    for dim, trait in enumerate(trait_names):
        # solid output line
        line, = ax_out.plot([], [], '-', linewidth=2.2, alpha=1.0,
                            label=f"{env_name} {trait}", zorder=2)
        lines[(env_name, dim)] = line
        # dotted target line on top
        target_val = ENVIRONMENTS[env_index]["target"][dim]
        tline, = ax_out.plot(range(GENERATIONS),
                             [target_val]*GENERATIONS, '--',
                             color=base_color, linewidth=1.0, alpha=0.6,
                             zorder=3, label=f"{env_name} {trait} target")
        target_lines[(env_name, dim)] = tline

ax_out.set_xlim(0, GENERATIONS)
ax_out.set_ylim(-0.15, 1)
ax_out.set_xlabel("Generation")
ax_out.set_ylabel("Best Brain Output")
ax_out.set_title("Best Brain Outputs")

# Legend 
legend = ax_out.legend(fontsize=9, loc='upper right', framealpha=0.9)

# --- Environment bands & emojis ---
num_stripes = 20
for (start, end, base_env_idx) in [(s, e, ENVIRONMENTS.index(b)) for (s, e, b) in env_spans]:
    base_env = ENVIRONMENTS[base_env_idx]
    next_env = ENVIRONMENTS[(base_env_idx + 1) % len(ENVIRONMENTS)]
    for i in range(num_stripes):
        x0 = start + (end - start) * (i / num_stripes)
        x1 = start + (end - start) * ((i + 1) / num_stripes)
        w = cosine_blend((i + 0.5) / num_stripes)
        c0 = np.array(to_rgb(env_colors[base_env["name"]]))
        c1 = np.array(to_rgb(env_colors[next_env["name"]]))
        c = (1 - w) * c0 + w * c1
        ax_out.axvspan(x0, x1, color=np.clip(c ** 0.7, 0, 1), alpha=0.35, linewidth=0)

    # emoji label at center
    center = (start + end) / 2
    ax_out.text(center, -0.06, env_emojis.get(base_env["name"], '?'),
                transform=ax_out.get_xaxis_transform(), ha='center', va='top', fontsize=16)

                



# --- Hidden activations ---
im_hid = ax_hid.imshow(np.zeros((HIDDEN_NEURONS, len(ENVIRONMENTS))), cmap='plasma', aspect='auto')
ax_hid.set_xticks(range(len(env_names))); ax_hid.set_xticklabels(env_names)
ax_hid.set_yticks(range(HIDDEN_NEURONS)); ax_hid.set_yticklabels([f'H{i}' for i in range(HIDDEN_NEURONS)])
ax_hid.set_title("Hidden Neuron Activations")
cbar_hid = fig.colorbar(im_hid, ax=ax_hid, fraction=0.046, pad=0.04)

# --- Weight heatmaps ---
im_w1 = ax_w1.imshow(np.zeros((3,HIDDEN_NEURONS)), vmin=-3, vmax=3, cmap="coolwarm")
ax_w1.set_title("Input→Hidden Weights"); fig.colorbar(im_w1, ax=ax_w1, fraction=0.046, pad=0.04)

im_w2 = ax_w2.imshow(np.zeros((HIDDEN_NEURONS,len(ENVIRONMENTS[0]["target"]))), vmin=-3, vmax=3, cmap="coolwarm")
ax_w2.set_title("Hidden→Output Weights"); fig.colorbar(im_w2, ax=ax_w2, fraction=0.046, pad=0.04)

# Slider 
ax_slider = plt.axes([0.22, 0.08, 0.58, 0.04])
slider = Slider(ax_slider, 'Generation', 0, GENERATIONS-1, valinit=GENERATIONS-1, valstep=1)

# ===== Style state  =====
style_state = {"thin": False, "markers": False, "trail": False}

def apply_style_to_lines():
    for line in lines.values():
        if style_state["thin"]:
            line.set_linewidth(1.5); line.set_alpha(0.7)
        else:
            line.set_linewidth(2.2); line.set_alpha(1.0)
        if style_state["markers"]:
            line.set_marker('o'); line.set_markersize(4)
        else:
            line.set_marker('')  
    for tline in target_lines.values():
        tline.set_alpha(0.5 if style_state["thin"] else 0.7)

def compute_x_range_for_gen(gen):
    if style_state["trail"]:
        start_idx = max(0, gen - TRAIL)
    else:
        start_idx = 0
    return start_idx, gen

def move_legend_dynamically(start_idx, end_idx):
    """
    In Trail mode, move the legend *away* from the visible window
    so it never covers what you're inspecting.
    """
    if style_state["trail"]:
        window_center = (start_idx + end_idx) / 2
        if window_center < GENERATIONS / 2:
            # looking near the start → move legend to the right
            legend.set_loc('upper right')
            legend.set_bbox_to_anchor((1.0, 1.0), transform=ax_out.transAxes)
        else:
            # looking near the end → move legend to the left
            legend.set_loc('upper left')
            legend.set_bbox_to_anchor((0.0, 1.0), transform=ax_out.transAxes)
    else:
        # Normal mode — fixed to upper right
        legend.set_loc('upper right')
        legend.set_bbox_to_anchor((1.0, 1.0), transform=ax_out.transAxes)

# Update function
def update(val):
    gen = int(slider.val)
    start_idx, end_idx = compute_x_range_for_gen(gen)

    # outputs (respect trail window)
    for env_name in env_names:
        for dim in range(len(trait_names)):
            ydata = best_outputs_history[env_name][dim][start_idx:end_idx+1]
            xdata = list(range(start_idx, end_idx+1))
            lines[(env_name, dim)].set_data(xdata, ydata)

    # hidden
    hid_data = np.array([best_hidden_history[env][gen] for env in env_names]).T
    im_hid.set_data(hid_data)
    im_hid.set_clim(hid_data.min(), hid_data.max())
    cbar_hid.update_normal(im_hid)

    # weights
    im_w1.set_data(best_w1_history[gen])
    im_w2.set_data(best_w2_history[gen])

    # styles + legend position
    apply_style_to_lines()
    move_legend_dynamically(start_idx, end_idx)

    fig.canvas.draw_idle()

slider.on_changed(update)

# --- Reset button ---
ax_reset = plt.axes([0.81, 0.08, 0.08, 0.04])
Button(ax_reset, 'Reset').on_clicked(lambda event: slider.reset())

# ===== CheckButtons: STYLE =====
rax_style = plt.axes([0.02, 0.60, 0.14, 0.25])  # left side
style_labels = ["Thin/Transparent", "Markers", "Trail mode"]
style_vis = [style_state["thin"], style_state["markers"], style_state["trail"]]
style_check = CheckButtons(rax_style, style_labels, style_vis)
rax_style.set_title("Style")

def style_toggle(label):
    if label == "Thin/Transparent":
        style_state["thin"] = not style_state["thin"]
    elif label == "Markers":
        style_state["markers"] = not style_state["markers"]
    elif label == "Trail mode":
        style_state["trail"] = not style_state["trail"]
    update(slider.val)

style_check.on_clicked(style_toggle)

# ===== CheckButtons: LINES (per Env×Trait) =====
rax_lines = plt.axes([0.02, 0.18, 0.14, 0.35])
line_labels = [f"{env}|{trait}" for env in env_names for trait in trait_names]  # safe delimiter
line_visibility = [True]*len(line_labels)
line_check = CheckButtons(rax_lines, line_labels, line_visibility)
rax_lines.set_title("Show/Hide Lines")

def line_toggle(label):
    # parse "Env|Trait"
    env, trait = label.split("|", 1)
    env = env.strip()
    trait = trait.strip()
    dim = trait_names.index(trait)
    key = (env, dim)
    visible = not lines[key].get_visible()
    lines[key].set_visible(visible)
    # keep target line in sync
    target_lines[key].set_visible(visible)
    fig.canvas.draw_idle()

line_check.on_clicked(line_toggle)

# initialize
update(GENERATIONS-1)
plt.show()
