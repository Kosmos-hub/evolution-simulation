# =========================
# Visualization
# =========================
from environments import ENVIRONMENTS, env_colors, env_spans
from utils import env_at_generation, add_env_bands
from evolve import TRAIL
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import to_rgb
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from creature_visuals import show_topk_creatures
from scipy.ndimage import gaussian_filter1d

def show_visualization(avg_fits, max_fits, stability, corr_history,
                       best_hidden_history, brains_outputs_history,
                       env_names, trait_names, TOP_K, GENERATIONS, HIDDEN_NEURONS):

    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(
        5, 2, figure=fig,
        width_ratios=[3.0, 1.2],
        height_ratios=[0.9, 1.0, 0.8, 1.3, 1.0]
    )
    plt.subplots_adjust(bottom=0.25, left=0.18, right=0.96, top=0.92, hspace=0.55)

    ax_out = fig.add_subplot(gs[:, 0])
    ax_hid = fig.add_subplot(gs[0, 1])     # top-right: activations
    ax_corr = fig.add_subplot(gs[1, 1])    # small middle-right: correlation
    ax_fit  = fig.add_subplot(gs[3:, 1])   # bottom-right: fitness/stability
    add_env_bands(ax_fit, alpha=0.07)
    add_env_bands(ax_corr, alpha=0.10)

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

    # --- Fitness history chart ---
    ax_fit.plot(
        range(GENERATIONS), avg_fits,
        label='Average Fitness',
        color='dodgerblue',
        linewidth=1.5,
        zorder=2
    )
    ax_fit.plot(
        range(GENERATIONS), max_fits,
        label='Max Fitness',
        color='crimson',
        linewidth=2.3,
        zorder=5,            # always drawn above other lines
        alpha=0.95
    )

    ax_fit.set_title("Population Fitness")
    ax_fit.set_xlabel("Generation")
    ax_fit.set_ylabel("Fitness")
    ax_fit.set_xlim(0, GENERATIONS)
    ax_fit.set_ylim(0, 1.05)
    ax_fit.grid(True, linestyle=(0, (2, 3)), alpha=0.3)
    ax_fit.legend(fontsize=8)


    # --- Best brain output stability ---
    ax_stab = ax_fit.twinx()

    # simple rolling mean (window=7) to reduce jitter
    win = 7
    kernel = np.ones(win) / win
    stab_smooth = np.convolve(stability, kernel, mode="same")

    ax_stab.plot(range(GENERATIONS), stab_smooth, color='purple', linewidth=1.5, alpha=0.9, label='Stability (std dev)')
    ax_stab.set_ylabel("Std Dev", color='purple')
    ax_stab.tick_params(axis='y', labelcolor='purple')

    # center colors at 0 for symmetric red/blue
    norm0 = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im_corr = ax_corr.imshow(np.zeros((3, 3)), norm=norm0, cmap="coolwarm")

    ax_corr.set_title("Trait Correlation (Best Brain)")
    ax_corr.set_xticks(range(len(trait_names))); ax_corr.set_xticklabels(trait_names)
    ax_corr.set_yticks(range(len(trait_names))); ax_corr.set_yticklabels(trait_names)
    # place colorbar neatly to the right of the correlation plot
    cbar_ax = fig.add_axes([ax_corr.get_position().x1 + 0.012,  # slight offset right
                            ax_corr.get_position().y0,
                            0.012,  # narrow width
                            ax_corr.get_position().height])
    fig.colorbar(im_corr, cax=cbar_ax)

    # labels inside the heatmap
    corr_texts = []
    def draw_corr_labels(mat):
        # clear old labels
        while corr_texts:
            corr_texts.pop().remove()
        # write numbers (skip diagonal)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i == j:  # dull the diagonal
                    continue
                t = ax_corr.text(j, i, f"{mat[i, j]:.2f}",
                                 ha="center", va="center", fontsize=8, color="black")
                corr_texts.append(t)

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
        # hide/show brain output lines
        for (k, env_name, dim), line in lines.items():
            if style_state["filter_env"]:
                line.set_visible(env_name == current_env_name and brain_visibility[k] and trait_visibility[k][dim])
            else:
                line.set_visible(brain_visibility[k] and trait_visibility[k][dim])

        # hide/show target reference lines too
        for (env_name, dim), tline in target_lines.items():
            if style_state["filter_env"]:
                tline.set_visible(env_name == current_env_name)
            else:
                tline.set_visible(True)

    def compute_x_range_for_gen(gen):
        if style_state["trail"]:
            return max(0, gen - TRAIL), gen
        else:
            return 0, gen
    
    def move_legend_dynamically(gen):
        # shift legend side depending on generation
        if gen < GENERATIONS / 2:
            legend.set_loc('upper right')
        else:
            legend.set_loc('upper left')



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

    # --- auto-highlight current environment region ---
    highlight_patch = ax_out.axvspan(0, 0, color='white', alpha=0.12, zorder=8)

    def update_highlight(gen):
        # remove old patch and draw a new one
        nonlocal highlight_patch
        highlight_patch.remove()
        for (start, end, base_env) in env_spans:
            if start <= gen < end:
                highlight_patch = ax_out.axvspan(start, end,
                                                 color=env_colors[base_env["name"]],
                                                 alpha=0.18, zorder=8)
                break

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

        # mask diagonal so it's visually de-emphasized
        mat = corr_history[gen].copy()
        mat = np.ma.array(mat, mask=np.eye(mat.shape[0], dtype=bool))
        im_corr.set_data(mat)
        draw_corr_labels(corr_history[gen])
        apply_style_to_lines()
        move_legend_dynamically(gen)
        update_highlight(gen)
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.22, 0.08, 0.58, 0.04])
    slider = Slider(ax_slider, 'Generation', 0, GENERATIONS-1, valinit=GENERATIONS-1, valstep=1)
    # slider.on_changed(update)
    ax_reset = plt.axes([0.81, 0.08, 0.08, 0.04])
    Button(ax_reset, 'Reset').on_clicked(lambda e: slider.reset())

    # --- Snapshot button ---
    ax_snap = plt.axes([0.90, 0.08, 0.08, 0.04])
    btn_snap = Button(ax_snap, 'Snapshot')

    def save_snapshot(event):
        gen = int(slider.val)
        # get current env name
        env_name = env_at_generation(gen)["name"].replace(" ", "_")
        filename = f"evolution_snapshot_gen{gen:03d}_{env_name}.png"
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Snapshot saved as {filename}")

    btn_snap.on_clicked(save_snapshot)

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
        
    # --- Create creature window linked to slider ---
    creature_fig, update_creatures = show_topk_creatures(
    brains_outputs_history, env_names, trait_names, TOP_K, GENERATIONS
    )

    def linked_update(val):
        gen = int(slider.val)
        update(val)            # existing update for graphs
        update_creatures(gen)  # new update for creatures

    slider.on_changed(linked_update)
    update_creatures(GENERATIONS - 1)

    from creature_visuals import show_milestone_creatures
    show_milestone_creatures(
        brains_outputs_history, env_names, trait_names, TOP_K, GENERATIONS
    )


    # initialize with generation 0 instead of the final one
    slider.set_val(0)
    update(0)
    update_creatures(0)

    # non-blocking display so all figures open together
    plt.show()
