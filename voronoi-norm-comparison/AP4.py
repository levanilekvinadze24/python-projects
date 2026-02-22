import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save PNGs without opening windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Plot directory ----------------------------------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Inline bridges instead of CSV
USE_CSV = False  # set True if you later want to read bridges.csv

# Edit this list to add bridges: ("Name", lat, lon)
BRIDGES = [
    ("White Bridge", 42.268741, 42.700420),
    ("Chain Bridge", 42.273804, 42.702360),
    ("Red Bridge",   42.271396, 42.699045),
    # ("Another Bridge", 42.27, 42.71),
]

def load_bridges(csv_path="bridges.csv"):
    """
    Returns names, lats, lons, from_csv_flag
    When USE_CSV is False, uses the inline BRIDGES list.
    """
    if USE_CSV and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {"name","lat","lon"}.issubset(df.columns):
            raise ValueError("bridges.csv must have columns: name,lat,lon")
        names = df["name"].astype(str).tolist()
        lats  = df["lat"].astype(float).to_numpy()
        lons  = df["lon"].astype(float).to_numpy()
        return names, lats, lons, True

    # Inline list
    names = [r[0] for r in BRIDGES]
    lats  = np.array([r[1] for r in BRIDGES], dtype=float)
    lons  = np.array([r[2] for r in BRIDGES], dtype=float)
    return names, lats, lons, False


# Optional SciPy: Euclidean Delaunay if available
_HAVE_SCIPY = False
try:
    from scipy.spatial import Delaunay
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# --- Local projection (meters)
def latlon_to_xy(lat, lon, lat0, lon0):
    """Simple local equirectangular projection around (lat0, lon0) in meters."""
    R = 6371000.0
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    lat0_r = math.radians(lat0); lon0_r = math.radians(lon0)
    x = R * (lon_r - lon0_r) * math.cos(lat0_r)
    y = R * (lat_r - lat0_r)
    return x, y

# --- Distances
def dist_matrix(grid_xy, pts_xy, metric="l2"):
    """grid_xy: (N,2), pts_xy: (M,2) -> (N,M) distances under metric in {l2,l1,linf}.FRom any place to bridge"""
    gx = grid_xy[:, 0][:, None]; gy = grid_xy[:, 1][:, None]
    px = pts_xy[:, 0][None, :];  py = pts_xy[:, 1][None, :]
    dx = gx - px; dy = gy - py
    if metric == "l2":   return np.hypot(dx, dy)
    if metric == "l1":   return np.abs(dx) + np.abs(dy)
    if metric == "linf": return np.maximum(np.abs(dx), np.abs(dy))
    raise ValueError("metric must be one of: l2, l1, linf")

# --- Grid Voronoi for any metric
def grid_voronoi(pts_xy, metric, bbox_xy, res=650):
    """Return Voronoi labels (res×res) by nearest site on a dense grid. each place to bridge"""
    xmin, xmax, ymin, ymax = bbox_xy
    xs = np.linspace(xmin, xmax, res); ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack([X.ravel(), Y.ravel()])
    D = dist_matrix(grid, pts_xy, metric=metric)
    labels = np.argmin(D, axis=1).reshape(res, res)
    extent = [xmin, xmax, ymin, ymax]
    return labels, extent

# --- Adjacency networks
def delaunay_edges(pts_xy):
    """Euclidean Delaunay edges if SciPy present; else fallback 3-NN under L2., find neigbours geomoetricaly"""
    M = pts_xy.shape[0]; edges = set()
    if _HAVE_SCIPY and M >= 3:
        tri = Delaunay(pts_xy)
        for s in tri.simplices:
            i, j, k = map(int, s)
            edges.update({tuple(sorted((i, j))), tuple(sorted((j, k))), tuple(sorted((i, k)))})
        return sorted(edges)
    # Fallback: 3-NN under L2
    G = dist_matrix(pts_xy, pts_xy, metric="l2")
    np.fill_diagonal(G, np.inf)
    for i in range(M):
        for j in np.argsort(G[i])[:3]:
            edges.add(tuple(sorted((i, int(j)))))
    return sorted(edges)

"""same , but with l1 linifinity"""
def knn_edges_metric(pts_xy, metric="l1", k=3):
    M = pts_xy.shape[0]; edges = set()
    G = dist_matrix(pts_xy, pts_xy, metric=metric)
    np.fill_diagonal(G, np.inf)
    for i in range(M):
        for j in np.argsort(G[i])[:k]:
            edges.add(tuple(sorted((i, int(j)))))
    return sorted(edges)

# --- Plotting
def plot_panel(ax, names, pts_xy, labels, extent, title, draw_delaunay=True,
               extra_edges=None, extra_label=None):
    M = len(names)
    colors = plt.cm.get_cmap("tab20", max(10, M)).colors[:M]
    ax.imshow(labels, extent=extent, origin="lower", cmap=ListedColormap(colors), alpha=0.55)
    ax.set_title(title)
    # points + labels
    ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=60, c="k", zorder=3)
    for i, n in enumerate(names):
        ax.text(pts_xy[i, 0], pts_xy[i, 1], f"  {n}", va="center", ha="left", fontsize=8, color="k")
    # Euclidean Delaunay
    if draw_delaunay:
        for i, j in delaunay_edges(pts_xy):
            ax.plot([pts_xy[i,0], pts_xy[j,0]], [pts_xy[i,1], pts_xy[j,1]], lw=1.2, c="k", alpha=0.75, zorder=2)
    # optional metric-specific edges
    if extra_edges:
        first = True
        for i, j in extra_edges:
            ax.plot(
                [pts_xy[i,0], pts_xy[j,0]],
                [pts_xy[i,1], pts_xy[j,1]],
                lw=1.0,
                c="white",
                alpha=0.9,
                ls="--",
                zorder=2,
                label=extra_label if first else None,
            )
            first = False
        if extra_label:
            ax.legend(loc="lower right", fontsize=7, frameon=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

# --- Area shares via grid ---------------------------------------------
def area_share(labels, extent, names):
    xmin, xmax, ymin, ymax = extent
    H, W = labels.shape
    dx = (xmax - xmin) / W; dy = (ymax - ymin) / H
    cell_area = dx * dy
    rows = []
    for i, nm in enumerate(names):
        count = int(np.sum(labels == i))
        rows.append({
            "metric": None,
            "bridge": nm,
            "area_m2": count * cell_area,
            "share": count / float(H*W),
            "pixels": count,
        })
    return pd.DataFrame(rows)

# --- Distance-to-nearest heatmap (optional visual) --------------------
def nearest_distance_map(pts_xy, bbox, res=650, metric="l2"):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, res); ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack([X.ravel(), Y.ravel()])
    D = dist_matrix(grid, pts_xy, metric=metric)
    dmin = D.min(axis=1).reshape(res, res)
    extent = [xmin, xmax, ymin, ymax]
    return dmin, extent

# --- Main --------------------------------------------------------------
def main():
    names, lats, lons, from_csv = load_bridges("bridges.csv")
    src = "bridges.csv" if from_csv else "inline list"
    print(f"[INFO] Using {src}, bridges: {len(names)}")

    lat0 = float(np.mean(lats)); lon0 = float(np.mean(lons))
    px, py = latlon_to_xy(lats, lons, lat0, lon0)
    pts_xy = np.column_stack([px, py])

    # bbox with padding
    xmin, xmax = px.min(), px.max(); ymin, ymax = py.min(), py.max()
    pad_x = 0.35 * (xmax - xmin if xmax > xmin else 1000.0)
    pad_y = 0.35 * (ymax - ymin if ymax > ymin else 1000.0)
    bbox = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    # Voronoi labels for each norm
    labels_l2, ext = grid_voronoi(pts_xy, "l2",   bbox, res=650)
    labels_l1, _   = grid_voronoi(pts_xy, "l1",   bbox, res=650)
    labels_li, _   = grid_voronoi(pts_xy, "linf", bbox, res=650)

    # metric-specific adjacency extras
    edges_l1 = knn_edges_metric(pts_xy, metric="l1", k=3)
    edges_li = knn_edges_metric(pts_xy, metric="linf", k=3)

    # combo figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    plot_panel(axs[0], names, pts_xy, labels_l2, ext,
               "Euclidean Voronoi + Euclidean Delaunay", draw_delaunay=True)
    plot_panel(
        axs[1],
        names,
        pts_xy,
        labels_l1,
        ext,
        "Manhattan Voronoi + Euclidean Delaunay\nwhite dashed = L1 kNN",
        draw_delaunay=True,
        extra_edges=edges_l1,
        extra_label="L1 kNN",
    )
    plot_panel(
        axs[2],
        names,
        pts_xy,
        labels_li,
        ext,
        "Chebyshev Voronoi + Euclidean Delaunay\nwhite dashed = L∞ kNN",
        draw_delaunay=True,
        extra_edges=edges_li,
        extra_label="L∞ kNN",
    )
    out_combo = os.path.join(PLOT_DIR, "kutaisi_bridges_voronoi_combo.png")
    plt.savefig(out_combo, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_combo}")

    # Individuals
    for fname, lab, ttl in [
        ("voronoi_euclidean.png",  labels_l2, "Euclidean Voronoi + Delaunay"),
        ("voronoi_manhattan.png",  labels_l1, "Manhattan Voronoi + Delaunay"),
        ("voronoi_chebyshev.png",  labels_li, "Chebyshev Voronoi + Delaunay"),
    ]:
        plt.figure(figsize=(6, 6))
        plot_panel(plt.gca(), names, pts_xy, lab, ext, ttl, draw_delaunay=True)
        fpath = os.path.join(PLOT_DIR, fname)
        plt.savefig(fpath, dpi=200)
        plt.close()
        print(f"[OK] Saved {fpath}")

    # Area tables per metric
    df_l2 = area_share(labels_l2, ext, names); df_l2["metric"] = "L2"
    df_l1 = area_share(labels_l1, ext, names); df_l1["metric"] = "L1"
    df_li = area_share(labels_li, ext, names); df_li["metric"] = "L∞"
    df_all = pd.concat(
        [df_l2, df_l1, df_li],
        ignore_index=True,
    )[["metric","bridge","area_m2","share","pixels"]]
    df_all.to_csv("bridge_area_by_metric.csv", index=False)
    print("[OK] Saved bridge_area_by_metric.csv")

    # Optional nearest-distance heatmap for L2
    dmin, ext2 = nearest_distance_map(pts_xy, bbox, res=650, metric="l2")
    plt.figure(figsize=(6,6))
    im = plt.imshow(dmin, extent=ext2, origin="lower", cmap="viridis")
    plt.colorbar(im, label="meters to nearest bridge")
    plt.scatter(pts_xy[:,0], pts_xy[:,1], c="white", s=30, edgecolors="black")
    for i,n in enumerate(names):
        plt.text(pts_xy[i,0], pts_xy[i,1], f"  {n}", color="white")
    plt.title("Distance to nearest bridge (Euclidean)")
    plt.tight_layout()
    heatmap_path = os.path.join(PLOT_DIR, "nearest_bridge_distance_heatmap_l2.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {heatmap_path}")

    print("\nDone. Swap BRIDGES list to use different sites, or set USE_CSV=True to read bridges.csv.")

if __name__ == "__main__":
    main()
