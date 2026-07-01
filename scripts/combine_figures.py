"""
combine_figures.py  —  merge result panels into side-by-side composites so the
paper fits in 5 pages (full-text-width, double-column float).

  Fig.\ results-A : policy 3D paths  |  policy CLF bars
  Fig.\ results-B : algo team return  |  resolution team return  |  CLF Pareto

Run AFTER the per-figure PNGs exist (compare_policies.py, compare_algos.py,
compare_resolutions.py, pareto_clf.py).  Re-run anytime to refresh.

    python combine_figures.py
"""
import os
from PIL import Image

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def hstack(srcs, dst, pad=12, bg=(255, 255, 255)):
    """Paste images left-to-right, scaled to a common height, onto white."""
    imgs = [Image.open(os.path.join(OUTDIR, s)).convert("RGB") for s in srcs]
    H = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * H / im.height), H)) for im in imgs]
    W = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + pad
    canvas.save(os.path.join(OUTDIR, dst))
    print(f"  wrote {dst}  ({W}x{H}) from {srcs}")


# Fig A: the headline policy comparison (paths + quantitative bars).
hstack(["policy_comparison_3d.png", "policy_comparison_bars.png"],
       "fig_results_A.png")

# Fig B: the three analysis panels (algorithm / resolution / weight Pareto).
hstack(["algo_return.png", "resolution_training_return.png", "pareto_clf.png"],
       "fig_results_B.png")

print("Done.")
