import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (no GUI windows)
import matplotlib.pyplot as plt

# ---------------------------------------
# Plot directory (all images go here)
# ---------------------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)



# 1D function f1 and exact derivative df1_exact at point x0
def f1(x: float) -> float:
    return math.exp(x) * math.sin(x)

def df1_exact(x: float) -> float:
    # exact derivative: e^x (sin x + cos x)
    return math.exp(x) * (math.sin(x) + math.cos(x))

# point where we evaluate derivative in 1D
x0 = 0.7

# 2D function f2 and exact partial derivatives at (x0_2d, y0_2d)
def f2(x: float, y: float) -> float:
    # example surface: x^2*y + sin(x*y) + y^3
    return (x * x) * y + math.sin(x * y) + (y ** 3)

def dfx2_exact(x: float, y: float) -> float:
    # exact ∂f/∂x
    return 2.0 * x * y + y * math.cos(x * y)

def dfy2_exact(x: float, y: float) -> float:
    # exact ∂f/∂y
    return (x * x) + x * math.cos(x * y) + 3.0 * (y ** 2)

# point where we evaluate gradient in 2D
x0_2d, y0_2d = 1.1, -0.8

# Optional: if True, will build f1,f2 and exact derivatives from SymPy expressions
USE_SYMPY = False
SYMPY_1D_EXPR = "exp(x)*sin(x)"
SYMPY_2D_EXPR = "x**2*y + sin(x*y) + y**3"


# =======================
#  FINITE DIFFERENCES
# =======================

# basic one-dimensional finite difference formulas
def diff_forward(f, x, h):  return (f(x + h) - f(x)) / h
def diff_backward(f, x, h): return (f(x) - f(x - h)) / h
def diff_central(f, x, h):  return (f(x + h) - f(x - h)) / (2.0 * h)
def diff_5point(f, x, h):   return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12.0 * h)

# central finite differences for gradient in 2D
def grad_central_2d(f, x, y, hx, hy):
    dfx = (f(x + hx, y) - f(x - hx, y)) / (2.0 * hx)
    dfy = (f(x, y + hy) - f(x, y - hy)) / (2.0 * hy)
    return dfx, dfy


# =======================
#  TANGENT / NORMAL HELPERS
# =======================

# line tangent to a 1D function at x0 with given slope
def tangent_line(f, x0, slope):
    f0 = f(x0)
    return lambda x: f0 + slope * (x - x0)

# plane tangent to a 2D surface at (x0,y0) with given partials fx, fy
def tangent_plane(f, x0, y0, fx, fy):
    f0 = f(x0, y0)
    return lambda x, y: f0 + fx * (x - x0) + fy * (y - y0)

# build normal vector from partial derivatives fx, fy
def normal_from_partials(fx, fy):
    # normal to z = f(x,y) is (-fx, -fy, 1)
    return np.array([-fx, -fy, 1.0], dtype=float)

# angle (in degrees) between two 3D vectors
def angle_between(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float)
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv == 0.0 or nw == 0.0:
        return np.nan
    c = np.clip(np.dot(v, w) / (nv * nw), -1.0, 1.0)
    return math.degrees(math.acos(c))


# =======================
#  Optional SymPy override
# =======================

def maybe_enable_sympy():
    """
    If USE_SYMPY is True, override f1, f2 and exact derivatives
    using symbolic expressions given above.
    """
    if not USE_SYMPY:
        return
    try:
        import sympy as sp
    except Exception:
        print("SymPy not available; continuing without it.")
        return

    # 1D case
    try:
        x = sp.symbols('x')
        expr = sp.sympify(SYMPY_1D_EXPR)
        dexpr = sp.diff(expr, x)
        f_num = sp.lambdify(x, expr, 'math')
        df_num = sp.lambdify(x, dexpr, 'math')
        globals()['f1'] = f_num
        globals()['df1_exact'] = df_num
        print(f"[SymPy] 1D expr OK: {SYMPY_1D_EXPR}")
    except Exception as e:
        print(f"[SymPy] 1D expr failed: {e}")

    # 2D case
    try:
        x, y = sp.symbols('x y')
        expr2 = sp.sympify(SYMPY_2D_EXPR)
        dfx = sp.diff(expr2, x)
        dfy = sp.diff(expr2, y)
        f2_num  = sp.lambdify((x, y), expr2,  'math')
        dfx_num = sp.lambdify((x, y), dfx,    'math')
        dfy_num = sp.lambdify((x, y), dfy,    'math')
        globals()['f2'] = f2_num
        globals()['dfx2_exact'] = dfx_num
        globals()['dfy2_exact'] = dfy_num
        print(f"[SymPy] 2D expr OK: {SYMPY_2D_EXPR}")
    except Exception as e:
        print(f"[SymPy] 2D expr failed: {e}")


# =======================
#  1D EXPERIMENT
# =======================

def run_1d_experiment():
    # test step sizes h from 1e-7 to 1e-1 (log scale)
    hs = np.logspace(-7, -1, 13)
    exact = df1_exact(x0)

    rows = []
    for h in hs:
        # absolute errors for each finite difference scheme
        e_fwd = abs(diff_forward(f1, x0, h)  - exact)
        e_bwd = abs(diff_backward(f1, x0, h) - exact)
        e_ctr = abs(diff_central(f1, x0, h)  - exact)
        e_5pt = abs(diff_5point(f1, x0, h)   - exact)
        rows.append([h, e_fwd, e_bwd, e_ctr, e_5pt])

    # save table of errors vs h
    df = pd.DataFrame(rows, columns=["h", "err_forward", "err_backward", "err_central", "err_5point"])
    df.to_csv("derivative_errors_f1.csv", index=False)

    # find best h (minimal error) for each method
    bests = {
        "forward":  ("err_forward",  df["err_forward"].idxmin()),
        "backward": ("err_backward", df["err_backward"].idxmin()),
        "central":  ("err_central",  df["err_central"].idxmin()),
        "5point":   ("err_5point",   df["err_5point"].idxmin()),
    }
    for name, (col, idx) in bests.items():
        row = df.loc[idx]
        print(f"1D best h [{name:8s}] = {row.h:.2e}  err={row[col]:.3e}")

    # log-log plot: error vs h for all methods
    plt.figure()
    plt.loglog(df["h"], df["err_forward"],  marker='o', label="Forward O(h)")
    plt.loglog(df["h"], df["err_backward"], marker='o', label="Backward O(h)")
    plt.loglog(df["h"], df["err_central"],  marker='o', label="Central O(h^2)")
    plt.loglog(df["h"], df["err_5point"],   marker='o', label="5-point O(h^4)")
    plt.xlabel("h")
    plt.ylabel("|approx - exact|")
    plt.title("1D derivative error vs h")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "derivative_errors_f1.png"), dpi=160)

    # plot function and tangent line near x0
    xs = np.linspace(x0 - 0.5, x0 + 0.5, 200)
    tline = tangent_line(f1, x0, exact)
    plt.figure()
    plt.plot(xs, [f1(x) for x in xs], label="f1(x)")
    plt.plot(xs, [tline(x) for x in xs], '--', label="tangent @ x0")
    plt.scatter([x0], [f1(x0)], s=30, label="x0")
    plt.title("Function vs tangent line (1D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "tangent_line_f1.png"), dpi=160)


# =======================
#  2D EXPERIMENT
# =======================

def run_2d_experiment():
    # test step sizes h from 1e-6 to 1e-2
    hs = np.logspace(-6, -2, 17)
    fx_exact = dfx2_exact(x0_2d, y0_2d)
    fy_exact = dfy2_exact(x0_2d, y0_2d)
    n_exact  = normal_from_partials(fx_exact, fy_exact)

    rows = []
    for h in hs:
        # central-difference gradient approximation at (x0_2d, y0_2d)
        fx_c, fy_c = grad_central_2d(f2, x0_2d, y0_2d, h, h)
        ex, ey = abs(fx_c - fx_exact), abs(fy_c - fy_exact)
        # normal from approximate gradient
        n_num = normal_from_partials(fx_c, fy_c)
        ang = angle_between(n_num, n_exact)
        # err_grad_L2 is combined gradient error
        rows.append([h, ex, ey, math.hypot(ex, ey), ang])

    # save 2D error table
    df = pd.DataFrame(rows, columns=["h", "err_fx", "err_fy", "err_grad_L2", "angle_error_deg"])
    df.to_csv("gradient_errors_f2.csv", index=False)

    # print best h based on gradient error and angle error
    i1 = df["err_grad_L2"].idxmin()
    i2 = df["angle_error_deg"].idxmin()
    print(f"2D best h (grad L2) = {df.loc[i1, 'h']:.2e}  err={df.loc[i1, 'err_grad_L2']:.3e}")
    print(f"2D best h (angle)   = {df.loc[i2, 'h']:.2e}  angle={df.loc[i2, 'angle_error_deg']:.4f} deg")

    # log-log plot: errors of fx, fy and full gradient vs h
    plt.figure()
    plt.loglog(df["h"], df["err_fx"],       marker='o', label="|fx - fx_exact|")
    plt.loglog(df["h"], df["err_fy"],       marker='o', label="|fy - fy_exact|")
    plt.loglog(df["h"], df["err_grad_L2"],  marker='o', label="||grad - grad*||_2")
    plt.xlabel("h")
    plt.ylabel("error")
    plt.title("2D gradient error vs h (central)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gradient_errors_f2.png"), dpi=160)

    # semilog plot: angle between true and approximate normal vs h
    plt.figure()
    plt.semilogx(df["h"], df["angle_error_deg"], marker='o')
    plt.xlabel("h")
    plt.ylabel("angle error (deg)")
    plt.title("Normal angle error vs h")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "normal_angle_error_f2.png"), dpi=160)

    # heatmap of |f - tangent plane| near (x0_2d, y0_2d)
    plane = tangent_plane(f2, x0_2d, y0_2d, fx_exact, fy_exact)
    xs = np.linspace(x0_2d - 0.4, x0_2d + 0.4, 40)
    ys = np.linspace(y0_2d - 0.4, y0_2d + 0.4, 40)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Zf = np.vectorize(f2)(X, Y)
    Zp = np.vectorize(plane)(X, Y)
    plt.figure()
    plt.imshow(
        np.abs(Zf - Zp),
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin='lower',
        aspect='auto',
    )
    plt.colorbar(label="|f - plane|")
    plt.title("Local linearization residual near (x0,y0)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "tangent_plane_residual_heatmap.png"), dpi=160)


# =======================
#  MAIN DRIVER
# =======================

def main():
    # optionally override f1,f2 and derivatives using SymPy expressions
    maybe_enable_sympy()

    print("Running Problem 3.1 experiments...")
    run_1d_experiment()
    run_2d_experiment()
    print("Done. Generated files:")
    print(" - derivative_errors_f1.csv, gradient_errors_f2.csv")
    print(" - plots/derivative_errors_f1.png, plots/tangent_line_f1.png")
    print(" - plots/gradient_errors_f2.png, plots/normal_angle_error_f2.png")
    print(" - plots/tangent_plane_residual_heatmap.png")
    print("\nSwap additional functions at the top and re-run if needed.")

if __name__ == "__main__":
    main()
