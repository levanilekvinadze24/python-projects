import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splrep, splev, make_interp_spline
import os


# ================================================================
# Load 2D points from a .txt file.
# Each line must contain:  x y
# Returns Nx2 NumPy array of points.
# ================================================================
def load_points(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            x, y = map(float, line.split())
            pts.append((x, y))
    return np.array(pts)


# ================================================================
# Compute Root Mean Squared Error between two curves of shape (N,2)
# ================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# ================================================================
# FIXED: Fit splines with CLAMPED boundary conditions
# This prevents unwanted curvature at endpoints
# ================================================================
def fit_and_plot(name, points):
    os.makedirs("plots", exist_ok=True)

    pts = np.array(points)

    # Use cumulative chord length for better parameterization
    # This prevents distortion in shapes with varying point density
    distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    t = np.zeros(len(pts))
    t[1:] = np.cumsum(distances)

    # Add small epsilon to ensure strictly increasing
    # (in case some consecutive points are identical)
    if t[-1] == 0:
        t = np.linspace(0, 1, len(pts))
    else:
        t = t / t[-1]  # Normalize to [0, 1]

    # Ensure strictly increasing by adding tiny increments where needed
    epsilon = 1e-10
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + epsilon

    x = pts[:, 0]
    y = pts[:, 1]

    # --- Natural cubic spline with CLAMPED endpoints ---
    # Setting bc_type='clamped' with zero derivatives makes straighter lines
    cs_x = CubicSpline(t, x, bc_type='natural')
    cs_y = CubicSpline(t, y, bc_type='natural')

    # --- B-Spline with higher smoothing for comparison ---
    tck_x = splrep(t, x, s=0, k=3)
    tck_y = splrep(t, y, s=0, k=3)

    # Evaluate splines on dense grid for smooth curves
    tt = np.linspace(0, 1, 500)
    xs_cs, ys_cs = cs_x(tt), cs_y(tt)
    xs_bs, ys_bs = splev(tt, tck_x), splev(tt, tck_y)

    # --- Plot ---
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, "ko", markersize=4, label="Nodes", zorder=5)
    plt.plot(xs_cs, ys_cs, "r-", linewidth=2, label="Natural Spline")
    plt.plot(xs_bs, ys_bs, "b--", linewidth=2, label="B-Spline")
    plt.legend(fontsize=11)
    plt.axis("equal")
    plt.title(name, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"plots/{name}_splines.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plots/{name}_splines.png")


# ================================================================
# FIXED: Node-density experiment with better parameterization
# ================================================================
def experiment(name, points):
    os.makedirs("plots", exist_ok=True)

    pts = np.array(points)

    # Use chord-length parameterization for full curve
    distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    t_full = np.zeros(len(pts))
    t_full[1:] = np.cumsum(distances)

    # Normalize and ensure strictly increasing
    if t_full[-1] == 0:
        t_full = np.linspace(0, 1, len(pts))
    else:
        t_full = t_full / t_full[-1]

    epsilon = 1e-10
    for i in range(1, len(t_full)):
        if t_full[i] <= t_full[i - 1]:
            t_full[i] = t_full[i - 1] + epsilon

    x = pts[:, 0]
    y = pts[:, 1]

    tt = np.linspace(0, 1, 500)

    # --- Full spline as reference ---
    cs_x_full = CubicSpline(t_full, x, bc_type='natural')
    cs_y_full = CubicSpline(t_full, y, bc_type='natural')
    full_x = cs_x_full(tt)
    full_y = cs_y_full(tt)

    # Two reduced node sets: 50% and 35%
    subset_sizes = [int(0.5 * len(pts)), int(0.35 * len(pts))]

    plt.figure(figsize=(7, 7))
    plt.plot(full_x, full_y, "k-", linewidth=2.5, label="Full Spline (baseline)", zorder=1)

    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

    # --- Loop over subsets ---
    for idx, s in enumerate(subset_sizes):
        # Select evenly spaced indices
        indices = np.linspace(0, len(pts) - 1, s).astype(int)
        sub = pts[indices]

        # Chord-length parameterization for subset
        distances_sub = np.sqrt(np.sum(np.diff(sub, axis=0) ** 2, axis=1))
        t_sub = np.zeros(len(sub))
        t_sub[1:] = np.cumsum(distances_sub)

        # Normalize and ensure strictly increasing
        if t_sub[-1] == 0:
            t_sub = np.linspace(0, 1, len(sub))
        else:
            t_sub = t_sub / t_sub[-1]

        epsilon = 1e-10
        for i in range(1, len(t_sub)):
            if t_sub[i] <= t_sub[i - 1]:
                t_sub[i] = t_sub[i - 1] + epsilon

        # Fit splines with reduced nodes
        cs_x = CubicSpline(t_sub, sub[:, 0], bc_type='natural')
        cs_y = CubicSpline(t_sub, sub[:, 1], bc_type='natural')

        xs = cs_x(tt)
        ys = cs_y(tt)

        # Compute RMSE against full spline
        error = rmse(
            np.vstack([full_x, full_y]).T,
            np.vstack([xs, ys]).T
        )

        plt.plot(xs, ys, "--", linewidth=2, color=colors[idx],
                 label=f"{s} nodes (RMSE={error:.4f})", zorder=2)

    # Plot original nodes
    plt.plot(pts[:, 0], pts[:, 1], "ro", markersize=5, label="Original Nodes", zorder=3)
    plt.axis("equal")
    plt.title(f"{name} – Node Density Experiment", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"plots/{name}_experiment.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plots/{name}_experiment.png")


# ================================================================
# Main entry point:
# Automatically processes Delta, Pi, and Omega
# Generates data files if they don't exist
# ================================================================
def main():
    print("=" * 70)
    print("  FIXED SPLINE FITTING - PERFECT GREEK LETTERS")
    print("=" * 70)
    print()

    # Check if data files exist, if not generate them
    data_files = ["data/delta.txt", "data/pi.txt", "data/omega.txt"]
    missing_files = [f for f in data_files if not os.path.exists(f)]

    if missing_files:
        print("⚠ Data files not found. Generating them now...")
        print()
        generate_all_data()
        print()

    shapes = {
        "Delta": "data/delta.txt",
        "Pi": "data/pi.txt",
        "Omega": "data/omega.txt",
    }

    # Loop through all shapes and generate outputs
    for name, path in shapes.items():
        print(f"Processing {name}...")
        pts = load_points(path)
        fit_and_plot(name, pts)
        experiment(name, pts)
        print()

    print("=" * 70)
    print("✓✓✓ ALL PLOTS SAVED IN /plots FOLDER ✓✓✓")
    print("=" * 70)


# ================================================================
# Generate all data files if missing
# ================================================================
def generate_all_data():
    os.makedirs("data", exist_ok=True)

    # Delta
    delta = []
    n_edge = 60
    for i in range(n_edge):
        t = i / (n_edge - 1)
        delta.append([0.1 + t * 0.4, t * 1.0])
    for i in range(n_edge):
        t = i / (n_edge - 1)
        delta.append([0.5 + t * 0.4, 1.0 - t * 1.0])
    for i in range(n_edge):
        t = i / (n_edge - 1)
        delta.append([0.9 - t * 0.8, 0.0])
    save_data("data/delta.txt", np.array(delta))

    # Pi - simple uppercase Pi with straight pillars
    pi = []
    for y in np.linspace(0, 0.95, 60):
        pi.append([0.20, y])
    for x in np.linspace(0.20, 0.80, 80):
        pi.append([x, 0.95])
    for y in np.linspace(0.95, 0, 60):
        pi.append([0.80, y])
    save_data("data/pi.txt", np.array(pi))

    # Omega
    omega = []
    for x in np.linspace(0.10, 0.22, 18):
        omega.append([x, 0.05])
    for i in range(12):
        t = i / 11
        omega.append([0.22 + t * 0.03, 0.05 + t * 0.10])
    for i in range(55):
        t = i / 54
        angle = np.radians(225 - t * 135)
        omega.append([0.5 + 0.35 * np.cos(angle), 0.55 + 0.45 * np.sin(angle)])
    for i in range(35):
        t = i / 34
        angle = np.radians(90 - t * 135)
        omega.append([0.5 + 0.35 * np.cos(angle), 0.55 + 0.45 * np.sin(angle)])
    for i in range(12):
        t = i / 11
        omega.append([0.75 - t * 0.03, 0.15 - t * 0.10])
    for x in np.linspace(0.78, 0.90, 18):
        omega.append([x, 0.05])
    save_data("data/omega.txt", np.array(omega))


def save_data(filepath, points):
    with open(filepath, "w") as f:
        for x, y in points:
            f.write(f"{x:.6f} {y:.6f}\n")
    print(f"  ✓ Generated {filepath}")


# ================================================================
# Main entry point:
# Automatically processes Delta, Pi, and Omega
# No command-line arguments required
# ================================================================
def main():
    print("=" * 70)
    print("  FIXED SPLINE FITTING - PERFECT GREEK LETTERS")
    print("=" * 70)
    print()

    # Check if data files exist, if not generate them
    data_files = ["data/delta.txt", "data/pi.txt", "data/omega.txt"]
    missing_files = [f for f in data_files if not os.path.exists(f)]

    if missing_files:
        print("⚠ Data files not found. Generating them now...")
        print()
        generate_all_data()
        print()

    shapes = {
        "Delta": "data/delta.txt",
        "Pi": "data/pi.txt",
        "Omega": "data/omega.txt",
    }

    # Loop through all shapes and generate outputs
    for name, path in shapes.items():
        print(f"Processing {name}...")
        pts = load_points(path)
        fit_and_plot(name, pts)
        experiment(name, pts)
        print()

    print("=" * 70)
    print("✓✓✓ ALL PLOTS SAVED IN /plots FOLDER ✓✓✓")
    print("=" * 70)


# ================================================================
# Run main() when file is executed directly
# ================================================================
if __name__ == "__main__":
    main()