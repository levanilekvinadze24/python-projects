import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os


# ============================================================
# 1) Load image, detect candle body profile (z, radius)
# ============================================================
def detect_profile(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image failed to load. Wrong format or corrupted file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove black background
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Canny edges
    edges = cv2.Canny(thresh, 80, 200)

    # Keep only largest connected component (the candle)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8,
    )
    if num_labels <= 1:
        raise ValueError("No edges detected.")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_edges = np.zeros_like(edges, dtype=np.uint8)
    clean_edges[labels == largest] = 255

    h, w = clean_edges.shape

    # --- width per row -> use it to drop flame rows ---
    widths = np.zeros(h, dtype=float)
    for y in range(h):
        xs = np.where(clean_edges[y, :] > 0)[0]
        if xs.size > 0:
            widths[y] = xs.max() - xs.min()

    max_width = widths.max()
    if max_width == 0:
        raise ValueError("No candle contour found.")

    # Keep only "wide" rows (candle body, not flame)
    body_rows = np.where(widths >= 0.3 * max_width)[0]
    if body_rows.size == 0:
        raise ValueError("No rows classified as candle body.")

    z_vals = []
    radii = []

    for y in body_rows:
        xs = np.where(clean_edges[y, :] > 0)[0]
        if xs.size < 2:
            continue

        left = xs.min()
        right = xs.max()
        width = right - left
        radius = width / 2.0  # half width is outer radius

        z_vals.append(y)
        radii.append(radius)

    z_vals = np.array(z_vals, dtype=float)
    radii = np.array(radii, dtype=float)

    if z_vals.size == 0:
        raise ValueError("No valid body points after filtering.")

    # Normalize height: 0 (top) .. 1 (bottom)
    z_vals = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())

    # Normalize radius: 0 .. 1
    radii = radii / radii.max()

    return z_vals, radii, img, clean_edges


# ============================================================
# 2) Fit spline r(z) to radius profile
# ============================================================
def fit_spline(z, r):
    idx = np.argsort(z)
    z_sorted, r_sorted = z[idx], r[idx]
    spline = UnivariateSpline(z_sorted, r_sorted, s=1e-4)
    return spline


# ============================================================
# Helper: radius profile (slight flattening at very top)
# ============================================================
def radius_profile(spline, z, flat_height=0.03):
    """
    spline : UnivariateSpline r(z)
    z      : scalar or array, in [0,1]
    flat_height : small band near the top with constant radius
    """
    z_arr = np.asarray(z, dtype=float)
    r = spline(z_arr)
    r = np.clip(r, 0.0, None)

    # keep radius constant in very thin band near top (vertical wall)
    r_flat = float(np.clip(spline(flat_height), 0.0, None))
    mask = z_arr <= flat_height
    r = np.where(mask, r_flat, r)

    if np.ndim(z) == 0:
        return float(r)
    return r


# ============================================================
# 3) Compute volume of body of revolution
# ============================================================
def compute_volume(spline, num=1000):
    z = np.linspace(0.0, 1.0, num)
    r = radius_profile(spline, z)
    integrand = np.pi * r**2
    vol = np.trapezoid(integrand, z)
    return float(vol)


# ============================================================
# 4) Create plots: edges, profile, 3D reconstruction
# ============================================================
def make_plots(img, edges, z, r_raw, spline):
    os.makedirs("plots", exist_ok=True)

    # ---------- Plot 1: Original + cleaned edges ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title("Canny edges (cleaned)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig("plots/01_edges.png", dpi=300)
    plt.close()

    # ---------- Plot 2: Radius profile & spline ----------
    z_s = np.linspace(0.0, 1.0, 400)
    r_s = radius_profile(spline, z_s)

    plt.figure(figsize=(6, 8))
    plt.plot(r_raw, z, "ko", markersize=3, label="Silhouette points")
    plt.plot(r_s, z_s, "r-", linewidth=2, label="Spline")

    plt.gca().invert_yaxis()
    plt.title("Candle radius profile")
    plt.xlabel("Normalized radius")
    plt.ylabel("Normalized height (z)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/02_profile.png", dpi=300)
    plt.close()

    # ---------- Plot 3: 3D reconstruction ----------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    theta = np.linspace(0, 2 * np.pi, 250)
    z_grid = np.linspace(0.0, 1.0, 250)
    theta, z_grid = np.meshgrid(theta, z_grid)

    r3d = radius_profile(spline, z_grid)
    r3d = np.clip(r3d, 0.0, None)

    X = r3d * np.cos(theta)
    Y = r3d * np.sin(theta)

    # ---- FLAT TOP CAP FOR SIDE SURFACE ----
    Z_side = z_grid.copy()
    flat_cap_height = 0.15   # top 15% becomes perfectly flat
    Z_side[z_grid <= flat_cap_height] = 0.0

    cmap = plt.get_cmap("Oranges")
    colors = cmap(0.3 + 0.7 * (1 - Z_side))

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")

    # side with flattened top
    ax.plot_surface(
        X, Y, Z_side,
        facecolors=colors,
        alpha=0.95,
        edgecolor="none",
        rstride=3,
        cstride=3,
    )

    # ---------- Top: outer flat ring + inner pressed bowl ----------
    tt = np.linspace(0, 2 * np.pi, 200)
    r_top = float(radius_profile(spline, 0.0))
    r_pool = 0.6 * r_top      # inner pool radius
    depression = 0.08         # depth of central pressed region

    # 1) Outer flat ring: r ∈ [r_pool, r_top], Z=0
    rr_ring = np.linspace(r_pool, r_top, 50)
    RR_ring, TT_ring = np.meshgrid(rr_ring, tt)
    X_ring = RR_ring * np.cos(TT_ring)
    Y_ring = RR_ring * np.sin(TT_ring)
    Z_ring = np.zeros_like(RR_ring)

    top_ring_col = np.array(cmap(0.95))
    ring_colors = np.repeat(top_ring_col[None, :], RR_ring.size, axis=0).reshape(
        RR_ring.shape + (4,),
    )
    ax.plot_surface(
        X_ring, Y_ring, Z_ring,
        facecolors=ring_colors,
        alpha=0.97,
        edgecolor="none",
    )

    # 2) Inner concave bowl: r ∈ [0, r_pool], pressed down
    rr_pool = np.linspace(0.0, r_pool, 60)
    RR_pool, TT_pool = np.meshgrid(rr_pool, tt)
    X_pool = RR_pool * np.cos(TT_pool)
    Y_pool = RR_pool * np.sin(TT_pool)

    # at r=r_pool -> Z=0; at r=0 -> Z=depression (deepest point)
    Z_pool = depression * (1.0 - (RR_pool / r_pool) ** 2)

    pool_col = np.array(cmap(0.8))
    pool_colors = np.repeat(pool_col[None, :], RR_pool.size, axis=0).reshape(
        RR_pool.shape + (4,),
    )
    ax.plot_surface(
        X_pool, Y_pool, Z_pool,
        facecolors=pool_colors,
        alpha=0.99,
        edgecolor="none",
    )

    # ---------- Bottom disk ----------
    rr_bot = np.linspace(0.0, float(radius_profile(spline, 1.0)), 80)
    RRb, TTb = np.meshgrid(rr_bot, tt)
    X_bot = RRb * np.cos(TTb)
    Y_bot = RRb * np.sin(TTb)
    Z_bot = np.ones_like(RRb)

    bot_col = np.array(cmap(0.3))
    bot_colors = np.repeat(bot_col[None, :], RRb.size, axis=0).reshape(
        RRb.shape + (4,),
    )
    ax.plot_surface(
        X_bot, Y_bot, Z_bot,
        facecolors=bot_colors,
        alpha=0.95,
        edgecolor="none",
    )

    ax.set_box_aspect((1, 1, 2))
    ax.view_init(elev=20, azim=40)

    r_max = float(r3d.max()) * 1.1
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(0.0, 1.05)

    ax.set_title("3D Reconstruction of Candle (Flat Top + Pressed Center)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (height)")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("plots/03_3d.png", dpi=300)
    plt.close()


# ============================================================
# 5) MAIN
# ============================================================
def main():
    image_path = "candle.jpg"   # your 2D candle image

    z, r, img, edges = detect_profile(image_path)
    spline = fit_spline(z, r)

    volume = compute_volume(spline)
    print(f"Estimated normalized volume (unitless): {volume:.4f}")

    make_plots(img, edges, z, r, spline)
    print("All plots saved in 'plots' folder.")


if __name__ == "__main__":
    main()
