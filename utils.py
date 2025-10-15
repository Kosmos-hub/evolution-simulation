import numpy as np
from environments import ENVIRONMENTS, env_spans, env_colors

# =========================
# Smooth environment blending (cosine)
# =========================
def cosine_blend(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

# =========================
# Environment lookup by generation
# =========================
def env_at_generation(gen):
    for i, (start, end, base_env) in enumerate(env_spans):
        if start <= gen < end:
            span_len = max(1, end - start)
            local_t = (gen - start) / span_len
            next_env = env_spans[(i + 1) % len(env_spans)][2]
            w = cosine_blend(local_t)

            # Delay reproduction transition a bit (shifted weight)
            w_repro = np.clip(w - 0.2, 0, 1)

            blend_inputs = [(1 - w) * base_env["inputs"][k] + w * next_env["inputs"][k] for k in range(3)]
            blend_targets = [
                (1 - w) * base_env["target"][0] + w * next_env["target"][0],  # Water
                (1 - w) * base_env["target"][1] + w * next_env["target"][1],  # Energy
                (1 - w_repro) * base_env["target"][2] + w_repro * next_env["target"][2]  # Reproduction (delayed)
            ]
            return {"name": base_env["name"], "inputs": blend_inputs, "target": blend_targets}


# =========================
# Environment band helpers
# =========================
def add_env_bands(ax, alpha=0.07):
    for (start, end, base_env) in env_spans:
        ax.axvspan(start, end, color=env_colors[base_env["name"]], alpha=alpha, zorder=-5)
