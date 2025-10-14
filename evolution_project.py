import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch

# === Configurable parameters ===
POPULATION_SIZE = 100 
GENERATIONS = 200
ELITISM = 2
MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 5
HIDDEN_NEURONS = 6
ENV_TESTS = 5

# disabled for randomness XD, remove "#" to keep same seed
# random.seed(1)
# np.random.seed(1)

# === Environments ===
ENVIRONMENTS = [
    {"name": "Daytime heatwave", "inputs": [1.0, 0.0, 0.0], "target": 0.1},
    {"name": "Cool night",       "inputs": [0.0, 0.0, 1.0], "target": 0.9},
    {"name": "Dry drought",      "inputs": [1.0, 1.0, 0.0], "target": 0.3},
]

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

# Build spans for background plotting
env_spans = []
cursor = 0
for base_env, duration in env_schedule:
    start = cursor
    end = min(cursor + duration, GENERATIONS)
    env_spans.append((start, end, base_env))
    cursor += duration

env_colors = {
    "Daytime heatwave": "#FFDD99",
    "Cool night": "#99CCFF",
    "Dry drought": "#FF9999",
}
env_emojis = {
    "Daytime heatwave": "☀️",
    "Cool night": "🌙",
    "Dry drought": "💧",
}

# === Neural network ===
class Brain:
    def __init__(self):
        self.w1 = np.random.randn(3, HIDDEN_NEURONS)
        self.w2 = np.random.randn(HIDDEN_NEURONS, 1)
    
    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.w1))
        o = 1 / (1 + np.exp(-np.dot(h, self.w2)))
        return o[0], h
    
    def mutate(self, rate=MUTATION_RATE):
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate

    def copy(self):
        new = Brain()
        new.w1 = np.copy(self.w1)
        new.w2 = np.copy(self.w2)
        return new

# === Environment variation for generation ===
def environment_for_generation(gen):
    gen_cursor = 0
    for base_env, duration in env_schedule:
        if gen_cursor <= gen - 1 < gen_cursor + duration:
            varied_inputs = [max(0.0, min(1.0, i + random.uniform(-0.5, 0.5)))
                             for i in base_env["inputs"]]
            varied_target = max(0.0, min(1.0, base_env["target"] + random.uniform(-0.3, 0.3)))
            return {"name": base_env["name"], "inputs": varied_inputs, "target": varied_target}
        gen_cursor += duration
    return ENVIRONMENTS[0]

# === Fitness ===
def fitness(brain):
    score = 0
    for _ in range(ENV_TESTS):
        env = random.choice(ENVIRONMENTS)
        varied_inputs = [max(0.0, min(1.0, i + random.uniform(-0.5,0.5))) for i in env["inputs"]]
        varied_target = max(0.0, min(1.0, env["target"] + random.uniform(-0.3,0.3)))
        output, _ = brain.forward(varied_inputs)
        score += 1 - abs(output - varied_target)
    return score / ENV_TESTS

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

# === Initialize population and stats ===
population = [Brain() for _ in range(POPULATION_SIZE)]
avg_fits = []
max_fits = []
best_outputs_history = {env['name']: [] for env in ENVIRONMENTS}
best_hidden_history = {env['name']: [] for env in ENVIRONMENTS}
best_w1_history = []
best_w2_history = []

env_names = [env['name'] for env in ENVIRONMENTS]

# === Evolution loop ===
for gen in range(1, GENERATIONS + 1):
    fits = [fitness(b) for b in population]
    avg_fits.append(np.mean(fits))
    max_fits.append(np.max(fits))

    best_idx = np.argmax(fits)
    best_brain = population[best_idx]

    for env_obj in ENVIRONMENTS:
        out, hidden = best_brain.forward(env_obj['inputs'])
        best_outputs_history[env_obj['name']].append(out)
        best_hidden_history[env_obj['name']].append(hidden.copy())
    
    best_w1_history.append(best_brain.w1.copy())
    best_w2_history.append(best_brain.w2.copy())

    # Evolve
    sorted_idx = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
    new_pop = [population[i].copy() for i in sorted_idx[:ELITISM]]
    while len(new_pop) < POPULATION_SIZE:
        parent = tournament_select(population, fits).copy()
        parent.mutate(rate=MUTATION_RATE)
        new_pop.append(parent)
    population = new_pop

# === Visualization ===
fig, axes = plt.subplots(2, 2, figsize=(18,10))
plt.subplots_adjust(bottom=0.25)
ax_out, ax_hid, ax_w1, ax_w2 = axes.flatten()

# Output lines
lines = {}
for env_name in env_names:
    line, = ax_out.plot([], [], label=env_name, linewidth=2)
    lines[env_name] = line
ax_out.set_xlim(0, GENERATIONS)
ax_out.set_ylim(-0.15, 1)
ax_out.set_xlabel("Generation")
ax_out.set_ylabel("Best Brain Output")
ax_out.set_title("Best Brain Outputs")
ax_out.legend(loc='upper right')

# Environment bands + emojis
for start, end, base_env in env_spans:
    color = env_colors.get(base_env['name'], "#CCCCCC")
    ax_out.axvspan(start, end, color=color, alpha=0.15, linewidth=0)
    center = (start + end)/2
    ax_out.text(center, -0.06, env_emojis.get(base_env['name'], '?'),
                transform=ax_out.get_xaxis_transform(),
                ha='center', va='top', fontsize=16)

# Hidden activations heatmap (dynamic)
im_hid = ax_hid.imshow(np.zeros((HIDDEN_NEURONS, len(ENVIRONMENTS))),
                       cmap='plasma', aspect='auto')
ax_hid.set_xticks(range(len(env_names)))
ax_hid.set_xticklabels(env_names)
ax_hid.set_yticks(range(HIDDEN_NEURONS))
ax_hid.set_yticklabels([f'H{i}' for i in range(HIDDEN_NEURONS)])
ax_hid.set_title("Hidden Neuron Activations")
cbar_hid = fig.colorbar(im_hid, ax=ax_hid)

# Weight heatmaps
im_w1 = ax_w1.imshow(np.zeros((3,HIDDEN_NEURONS)), vmin=-3, vmax=3, cmap="coolwarm")
ax_w1.set_title("Input→Hidden Weights")
ax_w1.set_xlabel("Hidden Neurons")
ax_w1.set_ylabel("Input Neurons")
fig.colorbar(im_w1, ax=ax_w1)

im_w2 = ax_w2.imshow(np.zeros((HIDDEN_NEURONS,1)), vmin=-3, vmax=3, cmap="coolwarm")
ax_w2.set_title("Hidden→Output Weights")
ax_w2.set_xlabel("Output Neuron")
ax_w2.set_ylabel("Hidden Neurons")
fig.colorbar(im_w2, ax=ax_w2)

# === Update function ===
def update(val):
    gen = int(slider.val)
    # outputs
    for env_name in env_names:
        ydata = best_outputs_history[env_name][:gen+1]
        xdata = list(range(len(ydata)))
        lines[env_name].set_data(xdata, ydata)
    # hidden activations
    hid_data = np.array([best_hidden_history[env][gen] for env in env_names]).T
    im_hid.set_data(hid_data)
    im_hid.set_clim(hid_data.min(), hid_data.max())
    cbar_hid.update_normal(im_hid)
    # weights
    im_w1.set_data(best_w1_history[gen])
    im_w2.set_data(best_w2_history[gen])
    fig.canvas.draw_idle()

# Slider for generation
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Generation', 0, GENERATIONS-1, valinit=GENERATIONS-1, valstep=1)
slider.on_changed(update)

# Reset button
ax_reset = plt.axes([0.83, 0.1, 0.08, 0.04])
Button(ax_reset, 'Reset').on_clicked(lambda event: slider.reset())

# Initialize visualization
update(GENERATIONS-1)
plt.show()
