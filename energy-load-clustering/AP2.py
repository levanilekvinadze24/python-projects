import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import os

from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------
#  Folder for all generated plots
# ----------------------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------
#  Synthetic data generator: 24h load profiles
# ----------------------------------------
def synth(n_days: int = 600, noise: float = 0.08, mix=(0.35, 0.45, 0.20), seed: int = 42) -> np.ndarray:
    # Reproducible random generator
    rng = np.random.default_rng(seed)

    # Hours of the day: 0..23
    h = np.arange(24)

    # Simple min-max normalization helper
    norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-8)

    # "Daytime" profile: peak around 13:00 + office hours effect
    day = norm(0.1 + 0.6*np.exp(-0.5*((h-13)/4)**2) + 0.2*(h>7)*(h<17))

    # "Evening" profile: strong peak around 20:00
    eve = norm(0.1 + 0.85*np.exp(-0.5*((h-20)/2.5)**2))

    # "Flat" profile: nearly constant with small sinusoidal variation
    flat = norm(0.2 + 0.05*np.sin(2*np.pi*h/24))

    # Number of days for each prototype (day / evening / flat)
    n_day, n_eve = int(n_days*mix[0]), int(n_days*mix[1])
    n_flat = n_days - n_day - n_eve

    def jitter(proto, m):
        # Random amplitude scaling per day
        amp = rng.uniform(0.7, 1.3, size=(m,1))
        # Add small local noise + global noise, clip negative values
        base = proto[None,:] * amp + rng.normal(0, 0.03, size=(m,24)) + rng.normal(0, noise, size=(m,24))
        return np.clip(base, 0, None)

    # Stack all days and shuffle the order
    X = np.vstack([jitter(day, n_day), jitter(eve, n_eve), jitter(flat, n_flat)])
    return X[rng.permutation(len(X))]

# ----------------------------------------
#  Norms / Distances
# ----------------------------------------
def _mat_l1(M: np.ndarray) -> float:
    # Matrix 1-norm (induced): maximum column sum of absolute values
    return np.max(np.sum(np.abs(M), axis=0))

def _mat_linf(M: np.ndarray) -> float:
    # Matrix infinity-norm (induced): maximum row sum of absolute values
    return np.max(np.sum(np.abs(M), axis=1))

def pairwise_dist(X: np.ndarray, norm: str = "l2", as_matrix: bool = False) -> np.ndarray:
    """
    Compute pairwise distance matrix for rows of X.

    norm:
      - "l2", "l1", "linf" → vector norms on R^24
      - "mat_l1", "mat_linf" → interpret each 24-dim row as 6x4 matrix
    as_matrix:
      - force matrix interpretation even if norm is not mat_*
    """
    n = X.shape[0]
    D = np.zeros((n,n), float)

    # Decide whether to treat each row as a 6x4 matrix
    use_mat = as_matrix or norm.startswith("mat_")
    Xm = X.reshape(-1,6,4) if use_mat else None

    for i in range(n):
        for j in range(i+1, n):
            if use_mat:
                # Matrix difference and matrix norm (L1 or Linf)
                diff = Xm[i] - Xm[j]
                d = _mat_l1(diff) if norm == "mat_l1" else _mat_linf(diff)
            else:
                # Vector difference and vector norm (L2, L1, or Linf)
                w = X[j] - X[i]
                d = np.linalg.norm(
                    w,
                    2 if norm=="l2" else (1 if norm=="l1" else np.inf),
                )
            D[i,j] = D[j,i] = d
    return D

# ----------------------------------------
#  Clustering algorithms
# ----------------------------------------
def fit_kmeans(X, k=3):
    # Standard K-Means clustering on feature space
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    y = km.fit_predict(X)
    # Return labels and cluster centers (24h prototypes)
    return y, km.cluster_centers_

def fit_dbscan(X, eps=1.2, min_samples=12, norm="l2", as_matrix=False):
    # Use a custom distance matrix (vector or matrix norm)
    D = pairwise_dist(X, norm=norm, as_matrix=as_matrix)
    # DBSCAN with precomputed distances; -1 labels will be noise points
    y = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples).fit_predict(D)
    return y, None  # DBSCAN has no explicit "centers"

def fit_optics(X, min_samples=12, xi=0.05, min_cluster_size=0.05, norm="l2", as_matrix=False):
    # Same idea as DBSCAN: precomputed distances, but using OPTICS
    D = pairwise_dist(X, norm=norm, as_matrix=as_matrix)
    y = OPTICS(
        metric="precomputed",
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    ).fit_predict(D)
    return y, None  # OPTICS also has no explicit centers

def fit_hier(X, k=3):
    # Agglomerative (hierarchical) clustering with Ward linkage
    y = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    # Compute cluster "centers" as mean profile in each cluster
    centers = np.vstack([X[y==c].mean(axis=0) for c in np.unique(y)])
    return y, centers

# ----------------------------------------
#  Evaluation / Summaries
# ----------------------------------------
def evaluate(X, y) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Silhouette and Davies-Bouldin only on non-noise points.

    For DBSCAN / OPTICS, label -1 is treated as noise and ignored.
    """
    m = y != -1
    # Need at least 2 points and 2 distinct clusters to evaluate
    if m.sum() <= 1 or len(np.unique(y[m])) <= 1:
        return None, None
    return silhouette_score(X[m], y[m]), davies_bouldin_score(X[m], y[m])

def summarize(X, y, centers=None):
    # Print cluster sizes (including possible noise label -1)
    u, c = np.unique(y, return_counts=True)
    print("cluster sizes:", dict(zip(u, c)))

    # Print mean profile (first 8 hours) for each real cluster
    for cl in [v for v in u if v!=-1]:
        mu = X[y==cl].mean(axis=0)
        print(f"C{cl} mean[0..7]:", np.round(mu[:8], 3))

    # If centers are available (KMeans / hierarchical), print prototype heads
    if centers is not None:
        print("prototypes[0..7]:", [np.round(c[:8],3) for c in centers])

# ----------------------------------------
#  Plotting with auto-save
# ----------------------------------------
def plot_centers(centers, title="prototypes", method="method"):
    import matplotlib.pyplot as plt

    # x-axis: hour of day
    h = np.arange(24)

    # Plot each cluster center as a 24h curve
    for i, c in enumerate(centers):
        plt.plot(h, c, label=f"C{i}")

    plt.xlabel("Hour")
    plt.ylabel("kWh (scaled)")
    plt.title(title)
    plt.legend()
    plt.xticks([0,6,12,18,23])
    plt.tight_layout()

    # Save figure into plots/centers_<method>.png
    fname = os.path.join(PLOT_DIR, f"centers_{method}.png")
    plt.savefig(fname, dpi=200)
    print("saved:", fname)

    plt.show()

def plot_pca(X, y, title="PCA(2)", method="method"):
    import matplotlib.pyplot as plt

    # Standardize features and project to 2D with PCA
    Z = StandardScaler().fit_transform(X)
    P = PCA(n_components=2, random_state=42).fit_transform(Z)

    # Scatter plot: each cluster with different color/label
    for cl in np.unique(y):
        m = y==cl
        plt.scatter(P[m,0], P[m,1], s=12, label=str(cl))

    plt.legend(title="cluster")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    # Save figure into plots/pca_<method>.png
    fname = os.path.join(PLOT_DIR, f"pca_{method}.png")
    plt.savefig(fname, dpi=200)
    print("saved:", fname)

    plt.show()

# ----------------------------------------
#  CLI entry point
# ----------------------------------------
def main():
    # Command line interface: configure data, method, and parameters
    ap = argparse.ArgumentParser(description="Cluster 24h load profiles")

    # Input data: CSV or synthetic if empty
    ap.add_argument("--csv", default="", help="CSV n x 24; if empty, synthetic data is used")

    # Clustering method choice
    ap.add_argument("--method", default="kmeans", choices=["kmeans","dbscan","optics","hier"])

    # KMeans / hierarchical: number of clusters
    ap.add_argument("--k", type=int, default=3)

    # DBSCAN parameters
    ap.add_argument("--eps", type=float, default=1.2)
    ap.add_argument("--min_samples", type=int, default=12)

    # OPTICS parameters
    ap.add_argument("--xi", type=float, default=0.05)
    ap.add_argument("--min_cluster_size", default=0.05)

    # Distance metric selection
    ap.add_argument("--norm", default="l2", choices=["l2","l1","linf","mat_l1","mat_linf"])

    # Treat each 24-dim point as 6x4 matrix (for matrix norms)
    ap.add_argument("--as_matrix", action="store_true")

    # Disable plotting if needed
    ap.add_argument("--no_plots", action="store_true")

    args = ap.parse_args()

    # Load data: CSV if provided, otherwise synthetic profiles
    X = pd.read_csv(args.csv).values.astype(float) if args.csv else synth()

    # Run selected clustering method
    if args.method == "kmeans":
        y, centers = fit_kmeans(X, k=args.k)
    elif args.method == "dbscan":
        y, centers = fit_dbscan(
            X,
            eps=args.eps,
            min_samples=args.min_samples,
            norm=args.norm,
            as_matrix=args.as_matrix,
        )
    elif args.method == "optics":
        y, centers = fit_optics(
            X,
            min_samples=args.min_samples,
            xi=args.xi,
            min_cluster_size=args.min_cluster_size,
            norm=args.norm,
            as_matrix=args.as_matrix,
        )
    else:
        # Hierarchical clustering
        y, centers = fit_hier(X, k=args.k)

    # Evaluate clustering quality (if possible) and print short summary
    sil, dbi = evaluate(X, y)
    summarize(X, y, centers)
    print(f"Silhouette={sil}  Davies-Bouldin={dbi}")

    # Optionally generate plots: cluster centers + PCA projection
    if not args.no_plots:
        if centers is not None:
            plot_centers(centers, f"{args.method} prototypes", method=args.method)
        plot_pca(X, y, f"{args.method} PCA(2)", method=args.method)

if __name__ == "__main__":
    main()
