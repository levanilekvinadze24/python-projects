"""
================================================================================
BATTERY THERMAL RUNAWAY SIMULATION - NEWTON-GAUSS-SEIDEL METHOD
================================================================================
This code implements:
1.  Initial Value Problem (IVP) for system of ODEs
2.  Implicit Euler numerical scheme
3.  Newton-Gauss-Seidel method
4.  Real-world battery thermal runaway problem
5.  Visualization of results
================================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


class BatteryModelNewtonGaussSeidel:
    """
    ============================================================================
    NEWTON-GAUSS-SEIDEL METHOD FOR BATTERY THERMAL RUNAWAY
    ============================================================================
    Solves the implicit Euler equations using Newton-Gauss-Seidel iteration:

    Implicit Euler: y_{n+1} = y_n + dt * f(y_{n+1}, I)
    Residual: G(y) = y - y_n - dt * f(y, I) = 0

    Newton-Gauss-Seidel iteration:
    For each component i: y_i^{k+1} = y_i^k - G_i(y^k) / (∂G_i/∂y_i)

    Where J_G = I - dt * J_f is the Jacobian of G
    ============================================================================
    """

    def __init__(self):
        """
        ========================================================================
        PHYSICAL PARAMETERS - IDENTICAL TO FIXED-POINT VERSION
        ========================================================================
        """
        # Thermal parameters
        self.m = 0.5  # Battery mass [kg]
        self.c_p = 1000.0  # Specific heat capacity [J/kg·K]
        self.h = 10.0  # Convective heat transfer coefficient [W/m²·K]
        self.A = 0.02  # Surface area [m²]
        self.T_amb = 298.0  # Ambient temperature [K] (25°C)

        # Electrochemical parameters
        self.Q_max = 3600.0 * 10.0  # Battery capacity [As] (10Ah)
        self.k_R = 0.0001  # Resistance growth coefficient
        self.T_ref = 298.0  # Reference temperature [K]

        # Arrhenius temperature dependence
        self.E_a = 3000.0  # Activation energy [J/mol]
        self.R_gas = 8.314  # Gas constant [J/mol·K]

        # Safety bounds (physical limits)
        self.T_min = 273.0  # Minimum temperature [K] (0°C)
        self.T_max = 400.0  # Maximum temperature [K] (127°C)
        self.SoC_min = 0.0  # Minimum state of charge (0%)
        self.SoC_max = 1.0  # Maximum state of charge (100%)
        self.R_min = 0.01  # Minimum internal resistance [Ω]
        self.R_max = 0.5  # Maximum internal resistance [Ω]

    def _clamp_state(self, y: np.ndarray) -> np.ndarray:
        """
        Ensure state variables stay within physical bounds.
        """
        y_clamped = y.copy()
        y_clamped[0] = max(self.T_min, min(self.T_max, y_clamped[0]))
        y_clamped[1] = max(self.SoC_min, min(self.SoC_max, y_clamped[1]))
        y_clamped[2] = max(self.R_min, min(self.R_max, y_clamped[2]))
        return y_clamped

    def _arrhenius_resistance(self, T: float, R: float) -> float:
        """
        Temperature-dependent resistance via Arrhenius equation.
        R_temp = R * exp[(E_a/R_gas) * (1/T_ref - 1/T)]
        """
        exponent = (self.E_a / self.R_gas) * (1.0 / self.T_ref - 1.0 / T)
        exponent = max(-10.0, min(10.0, exponent))  # Prevent overflow
        R_temp = R * float(np.exp(exponent))
        return min(R_temp, 1.0)  # Cap at 1.0 Ω for stability

    def f(self, y: np.ndarray, I: float) -> np.ndarray:
        """
        ========================================================================
        RIGHT-HAND SIDE OF ODE SYSTEM - SAME AS FIXED-POINT VERSION
        ========================================================================
        Identical ODE system to ensure fair comparison.

        System of 3 ordinary differential equations:
        1. Temperature: dT/dt = (Q_gen - Q_loss) / (m * c_p)
        2. State of Charge: dSoC/dt = -|I| / Q_max
        3. Internal Resistance: dR/dt = k_R * (T - T_ref) * (1 - SoC)
        ========================================================================
        """
        y = self._clamp_state(y)
        T, SoC, R = float(y[0]), float(y[1]), float(y[2])

        # Temperature-dependent resistance
        R_temp = self._arrhenius_resistance(T, R)

        # Heat generation and loss
        Q_gen = (abs(I) ** 2) * R_temp  # Joule heating
        Q_loss = self.h * self.A * (T - self.T_amb)  # Convective cooling

        # System of ODEs
        dT_dt = (Q_gen - Q_loss) / (self.m * self.c_p)
        dSoC_dt = -abs(I) / self.Q_max
        dR_dt = self.k_R * (T - self.T_ref) * (1.0 - SoC)

        return np.array([dT_dt, dSoC_dt, dR_dt], dtype=float)

    def jacobian(self, y: np.ndarray, I: float, eps: float = 1e-7) -> np.ndarray:
        """
        ========================================================================
        NUMERICAL JACOBIAN COMPUTATION
        ========================================================================
        Computes J_f = ∂f/∂y using finite differences.
        Required for Newton-Gauss-Seidel method.

        J_f[i,j] = [f_i(y + ε·e_j) - f_i(y)] / ε

        Where e_j is the j-th unit vector.
        ========================================================================
        """
        n = len(y)
        J = np.zeros((n, n), dtype=float)
        f0 = self.f(y, I)  # f at current y

        for j in range(n):
            y2 = y.copy()
            y2[j] += eps  # Perturb j-th component
            fj = self.f(y2, I)
            J[:, j] = (fj - f0) / eps  # Finite difference

        return J

    def newton_gauss_seidel(self, y_n: np.ndarray, dt: float, I: float,
                            tol: float = 1e-6, max_iter: int = 50):
        """
        ========================================================================
        NEWTON-GAUSS-SEIDEL METHOD IMPLEMENTATION
        ========================================================================
        Solves: G(y) = y - y_n - dt * f(y, I) = 0

        Using Newton-Gauss-Seidel iteration:
        1. Start: y^0 = y_n
        2. For each iteration k:
           a. Compute J_f(y^k) and J_G = I - dt * J_f
           b. For each component i:
              y_i^{k+1} = y_i^k - G_i(y^k) / J_G[i,i]
        3. Stop when ||y^{k+1} - y^{k}|| < tol

        Returns: Solution y_{n+1} and iteration count
        ========================================================================
        """
        y_new = y_n.copy()  # Initial guess

        for k in range(max_iter):
            y_old = y_new.copy()

            # Compute Jacobian of f at current iterate
            J_f = self.jacobian(y_new, I)

            # Jacobian of G: J_G = I - dt * J_f
            J_G = np.eye(len(y_new)) - dt * J_f

            # Gauss-Seidel component updates
            for i in range(len(y_new)):
                # Residual G(y) = y - y_n - dt * f(y, I)
                G = y_new - y_n - dt * self.f(y_new, I)

                # Newton-Gauss-Seidel update using diagonal entry
                denom = float(J_G[i, i])
                if abs(denom) > 1e-10:  # Avoid division by zero
                    y_new[i] = y_new[i] - float(G[i]) / denom

            # Apply physical constraints
            y_new = self._clamp_state(y_new)

            # Convergence check
            error = float(np.linalg.norm(y_new - y_old))
            if error < tol:
                return y_new, (k + 1)

        return y_new, max_iter  # Return best effort if not converged

    def solve(self, t_span, y0, I_func, dt: float = 0.5, tol: float = 1e-6):
        """
        ========================================================================
        TIME-MARCHING SOLVER - IMPLICIT EULER WITH NEWTON-GAUSS-SEIDEL
        ========================================================================
        Solves the IVP over time interval t_span.

        Parameters:
        - t_span: (t_start, t_end) time interval
        - y0: Initial condition [T0, SoC0, R0]
        - I_func: Function I(t) returning current at time t
        - dt: Time step size
        - tol: Convergence tolerance for Newton-Gauss-Seidel

        Returns: Dictionary with solution and performance metrics
        ========================================================================
        """
        t0, t1 = float(t_span[0]), float(t_span[1])
        t = np.arange(t0, t1 + dt, dt, dtype=float)
        n_steps = len(t)

        # Initialize solution array
        y = np.zeros((n_steps, 3), dtype=float)
        y[0] = self._clamp_state(np.array(y0, dtype=float))

        # Performance tracking
        total_iters = 0
        iter_counts = []
        start = time.time()

        # Time marching loop
        for i in range(n_steps - 1):
            I = float(I_func(t[i + 1]))  # Current at next time step

            # Solve implicit step using Newton-Gauss-Seidel
            y[i + 1], iters = self.newton_gauss_seidel(y[i], dt, I, tol=tol)
            total_iters += iters
            iter_counts.append(iters)

            # Stop if battery is fully discharged
            if y[i + 1, 1] <= 0.0:
                y[i + 1:, 1] = 0.0
                break

        # Performance metrics
        cpu_time = time.time() - start
        avg_iters = total_iters / max(1, (n_steps - 1))

        return {
            "t": t,
            "T": y[:, 0],  # Temperature [K]
            "SoC": y[:, 1],  # State of Charge [0-1]
            "R": y[:, 2],  # Internal Resistance [Ω]
            "cpu_time": cpu_time,
            "total_iterations": total_iters,
            "avg_iterations": avg_iters,
            "min_iterations": int(min(iter_counts)) if iter_counts else 0,
            "max_iterations": int(max(iter_counts)) if iter_counts else 0,
        }


# ==============================================================================
# CURRENT PROFILE FUNCTIONS (IDENTICAL TO FIXED-POINT VERSION)
# ==============================================================================
def current_profile_normal(t: float) -> float:
    """Normal charging: 2C rate (20A for 10Ah battery)"""
    return 20.0


def current_profile_fast(t: float) -> float:
    """Fast charging: 5C rate (50A for 10Ah battery)"""
    return 50.0


def current_profile_extreme(t: float) -> float:
    """Extreme pulsed charging profile"""
    if t < 500.0:
        return 70.0
    if t < 1000.0:
        return 35.0
    return 70.0


# ==============================================================================
# UTILITY FUNCTIONS (IDENTICAL TO FIXED-POINT VERSION)
# ==============================================================================
def ensure_output_dir(dir_name: str) -> str:
    """Create output directory if it doesn't exist"""
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_figure(fig, out_dir: str, filename: str):
    """Save matplotlib figure to file"""
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


# ==============================================================================
# MAIN EXECUTION - NEWTON-GAUSS-SEIDEL METHOD ONLY
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BATTERY THERMAL RUNAWAY - NEWTON-GAUSS-SEIDEL METHOD")
    print("=" * 80)

    # Create output directory
    out_dir = ensure_output_dir("battery_results_newton")

    # Initialize model
    model = BatteryModelNewtonGaussSeidel()

    # Initial conditions: [Temperature(K), SoC(0-1), Resistance(Ω)]
    y0 = np.array([298.0, 1.0, 0.05], dtype=float)  # 25°C, 100% SoC, 0.05Ω

    # Time span: 0 to 1800 seconds (30 minutes)
    t_span = (0.0, 1800.0)

    # Time step size
    dt = 0.5  # seconds

    # Charging scenarios (identical to Fixed-Point)
    scenarios = [
        ("Normal Charging (2C)", current_profile_normal),
        ("Fast Charging (5C)", current_profile_fast),
        ("Extreme Fast Charging (Pulsed)", current_profile_extreme),
    ]

    results_all = {}

    # Run simulations for all scenarios
    for name, I_func in scenarios:
        print(f"\n{name}")
        r = model.solve(t_span, y0, I_func, dt=dt, tol=1e-6)
        results_all[name] = r

        # Print performance metrics
        print(f"  CPU Time:           {r['cpu_time']:.4f} s")
        print(f"  Total Iterations:   {r['total_iterations']}")
        print(f"  Avg Iter/Step:      {r['avg_iterations']:.2f}")
        print(f"  Min Iter/Step:      {r['min_iterations']}")
        print(f"  Max Iter/Step:      {r['max_iterations']}")
        print(f"  Max Temperature:    {np.max(r['T']) - 273.0:.2f} °C")
        print(f"  Final SoC:          {r['SoC'][-1]:.4f}")
        print(f"  Final Resistance:   {r['R'][-1] * 1000:.2f} mΩ")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Battery Thermal Runaway Dynamics - Newton-Gauss-Seidel",
                 fontsize=14, fontweight="bold")

    colors = ["blue", "orange", "red"]

    # Plot 1: Temperature Evolution
    ax = axes[0, 0]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["T"] - 273.0, label=name, color=colors[idx], linewidth=2)
    ax.axhline(y=60.0, color="red", linestyle="--", alpha=0.5, label="Safety Limit (60°C)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Evolution", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: State of Charge Depletion
    ax = axes[0, 1]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["SoC"] * 100.0, label=name, color=colors[idx], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("State of Charge Depletion", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Internal Resistance Growth
    ax = axes[1, 0]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["R"] * 1000.0, label=name, color=colors[idx], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Internal Resistance (mΩ)")
    ax.set_title("Internal Resistance Growth", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Phase Portrait (T vs SoC)
    ax = axes[1, 1]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["SoC"] * 100.0, r["T"] - 273.0, label=name, color=colors[idx], linewidth=2)
        ax.plot(r["SoC"][0] * 100.0, r["T"][0] - 273.0, "o", color=colors[idx], markersize=8)
    ax.set_xlabel("State of Charge (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Phase Portrait (T vs SoC)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, out_dir, "battery_newton_results.png")
    plt.show()

    print("\n" + "=" * 80)
    print("NEWTON-GAUSS-SEIDEL METHOD COMPLETED SUCCESSFULLY")
    print("=" * 80)