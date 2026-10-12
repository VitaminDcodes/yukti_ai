"""
paint_simulation/report.py

Coating quality analytics and thickness distribution heatmap generation.
Exports visual inspection maps and industrial coating evaluation metrics.
"""

import numpy as np


def save_thickness_heatmap(painter, filepath: str = "docs/paint_deposition_heatmap.png"):
    """
    Renders and saves a 2D contour heatmap of dry film thickness (DFT) across the car body.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[REPORT] matplotlib not installed; skipping heatmap export.")
        return

    thickness_map = painter.thickness_map  # [res_z, res_x] in microns
    metrics = painter.get_metrics()

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    # Extent: [x_min, x_max, z_min, z_max]
    extent = [painter.x_min, painter.x_max, painter.z_min, painter.z_max]

    im = ax.imshow(
        thickness_map,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(60.0, float(metrics["max_thickness_um"]))
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Dry Film Thickness (DFT) [$\\mu m$]", fontsize=11, fontweight="bold")

    ax.set_title(
        f"Automotive Spray Painting Simulation - Coating Thickness Distribution\n"
        f"Coverage: {metrics['target_coverage_pct']:.1f}% | "
        f"Mean Thickness: {metrics['mean_thickness_um']:.1f} $\\mu m$ | "
        f"Uniformity: {metrics['uniformity_pct']:.1f}% | "
        f"Paint Used: {metrics['paint_sprayed_ml']:.1f} mL",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Car Longitudinal Axis X [m]", fontsize=11)
    ax.set_ylabel("Car Vertical Axis Z [m]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[REPORT] Paint deposition heatmap saved to '{filepath}'.")
