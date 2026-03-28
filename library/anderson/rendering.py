from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def draw_cluster(ax, positions: np.ndarray, title: str, bond_threshold: float = 1.55, cmap: str = "viridis") -> None:
    pos = np.asarray(positions, dtype=float)
    ax.set_title(title)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=60,
        c=np.linspace(0.1, 0.9, len(pos)),
        cmap=cmap,
        edgecolors="black",
        linewidths=0.4,
    )
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if np.linalg.norm(pos[i] - pos[j]) <= bond_threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="gray",
                    alpha=0.4,
                    lw=1.1,
                )
    lim = max(2.8, float(np.max(np.abs(pos))) + 0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_cluster_result(initial_positions: np.ndarray, final_positions: np.ndarray, trace: list[float], title: str) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2])
    ax_init = fig.add_subplot(gs[0, 0], projection="3d")
    ax_final = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, :])

    draw_cluster(ax_init, initial_positions, "Geometria estrutural")
    draw_cluster(ax_final, final_positions, "Geometria final")
    ax_trace.plot(np.asarray(trace, dtype=float), color="tab:blue", lw=2.0)
    ax_trace.set_title("Trajetoria da energia")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig
