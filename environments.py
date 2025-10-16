import random

# =========================
# Configurable environment definitions
# =========================
GENERATIONS = 200
MIN_DURATION = 20
MAX_DURATION = 40

ENVIRONMENTS = [
    {"name": "Daytime heatwave", "inputs": [1.0, 0.0, 0.0], "target": [0.4, 0.7, 0.2]},
    {"name": "Cool night",        "inputs": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    {"name": "Dry drought",       "inputs": [1.0, 1.0, 0.0], "target": [0.8, 0.7, 0.0]},
    {"name": "Humid twilight",    "inputs": [0.5, 0.3, 0.7], "target": [0.2, 0.5, 0.8]},
    # target trait outputs: [water retention, energy effiency, reproduction]
]

# =========================
# Environment schedule and spans
# =========================
env_schedule = []
gen_count = 0
while gen_count < GENERATIONS:
    for base_env in ENVIRONMENTS:
        duration = random.randint(MIN_DURATION, MAX_DURATION)
        env_schedule.append((base_env, duration))
        gen_count += duration
        if gen_count >= GENERATIONS:
            break

env_spans = []
cursor = 0
for base_env, duration in env_schedule:
    start = cursor
    end = min(cursor + duration, GENERATIONS)
    env_spans.append((start, end, base_env))
    cursor += duration

# =========================
# Visual helpers
# =========================
env_colors = {
    "Daytime heatwave": "#FFDD99",
    "Cool night": "#99CCFF",
    "Dry drought": "#FF9999",
    "Humid twilight": "#CC99FF",
}

env_emojis = {
    "Daytime heatwave": "☀️",
    "Cool night": "🌙",
    "Dry drought": "💧",
    "Humid twilight": "🌫️",
}
# im only keeping them in because the code breaks without them ¯\_(ツ)_/¯
