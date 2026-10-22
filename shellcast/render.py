"""Quick shaded renders (matplotlib, no GPU) for before/after previews of sculpt edits."""

from __future__ import annotations

import numpy as np


def _draw(ax, t, view, i, j, title, flip=1):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    n = t.face_normals
    front = (n @ view) > 0
    tri = t.triangles[front]
    order = np.argsort(tri.mean(1) @ view)
    light = np.array([0.3, -0.6, 0.75]); light /= np.linalg.norm(light)
    L = view * 0.6 + light * 0.4; L /= np.linalg.norm(L)
    shade = np.clip(0.25 + 0.75 * np.abs(n[front] @ L), 0, 1)
    polys = tri[order][:, :, [i, j]].copy()
    polys[:, :, 0] *= flip
    ax.add_collection(PolyCollection(polys, facecolors=plt.cm.gray(shade[order]), edgecolors="none"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def render_views(before, after, path, crop=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    _draw(axes[0, 0], before, np.array([0, -1, 0]), 0, 2, "before · front")
    _draw(axes[0, 1], after, np.array([0, -1, 0]), 0, 2, "after · front")
    _draw(axes[1, 0], before, np.array([1, 0, 0]), 1, 2, "before · side", flip=-1)
    _draw(axes[1, 1], after, np.array([1, 0, 0]), 1, 2, "after · side", flip=-1)
    if crop:
        (x0, y0, z0), (x1, y1, z1) = crop
        for ax in axes.ravel():
            ax.set_ylim(z0 - 40, z1 + 40)
    fig.patch.set_facecolor("#0b0f1a")
    for ax in axes.ravel():
        ax.set_facecolor("#0b0f1a"); ax.title.set_color("#9aa3b2")
    plt.tight_layout(); plt.savefig(path, dpi=55); plt.close(fig)
