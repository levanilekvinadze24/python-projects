"""
================================================================================
BATTERY THERMAL RUNAWAY SIMULATION - FIXED-POINT ITERATION METHOD
================================================================================
This code implements:
1.  Initial Value Problem (IVP) for system of ODEs
2.  Implicit Euler numerical scheme
3.  Fixed-Point Iteration method
4. Real-world battery thermal runaway problem
5. Visualization of results
================================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


class BatteryModelFixedPoint:
    """
    ============================================================================
    FIXED-POINT ITERATION METHOD FOR BATTERY THERMAL RUNAWAY
    ============================================================================
    Goal:
    - Simulate battery heating + discharge + resistance growth (3 state variables)
    - Use Implicit Euler time stepping (stable for stiff problems)
    - Solve the implicit step with Fixed-Point iteration

    State vector:
      y = [T, SoC, R]
      T   = temperature (K)
      SoC = state of charge (0..1)
      R   = internal resistance (Ohm)

    Implicit Euler step:
      y_{n+1} = y_n + dt * f(y_{n+1}, I)

    Fixed-point iteration idea:
      start guess y^0 = y_n
      repeat: y^{k+1} = y_n + dt * f(y^k, I)
      until change is small
    ============================================================================
    """

    def __init__(self):
        """
        Store all physical constants and safety limits used by the model.
        """
        # -------------------- THERMAL PARAMETERS --------------------
        self.m = 0.5          # battery mass [kg]
        self.c_p = 1000.0     # specific heat capacity [J/(kg*K)]
        self.h = 10.0         # convective heat transfer coefficient [W/(m^2*K)]
        self.A = 0.02         # battery surface area [m^2]
        self.T_amb = 298.0    # ambient temperature [K] (25°C)

        # -------------------- ELECTROCHEMICAL PARAMETERS --------------------
        self.Q_max = 3600.0 * 10.0  # capacity [As] (10Ah)
        self.k_R = 0.0001           # resistance growth coefficient
        self.T_ref = 298.0          # reference temperature [K]

        # -------------------- ARRHENIUS PARAMETERS (temperature effect on R) --------------------
        self.E_a = 3000.0    # activation energy [J/mol]
        self.R_gas = 8.314   # gas constant [J/(mol*K)]

        # -------------------- SAFETY / PHYSICAL LIMITS (avoid nonsense values) --------------------
        self.T_min = 273.0   # 0°C in Kelvin
        self.T_max = 400.0   # ~127°C in Kelvin
        self.SoC_min = 0.0
        self.SoC_max = 1.0
        self.R_min = 0.01
        self.R_max = 0.5

    def _clamp_state(self, y: np.ndarray) -> np.ndarray:
        """
        Keep T, SoC, R inside safe bounds so the simulation doesn't explode.
        This is NOT a numerical method step; it's just safety clipping.
        """
        y_clamped = y.copy()

        # Clamp temperature
        y_clamped[0] = max(self.T_min, min(self.T_max, y_clamped[0]))

        # Clamp state of charge
        y_clamped[1] = max(self.SoC_min, min(self.SoC_max, y_clamped[1]))

        # Clamp resistance
        y_clamped[2] = max(self.R_min, min(self.R_max, y_clamped[2]))

        return y_clamped

    def _arrhenius_resistance(self, T: float, R: float) -> float:
        """
        Make resistance depend on temperature using a simple Arrhenius-like factor.

        R_temp = R * exp[(E_a/R_gas) * (1/T_ref - 1/T)]

        We also cap exponent and the final result to keep it numerically stable.
        """
        exponent = (self.E_a / self.R_gas) * (1.0 / self.T_ref - 1.0 / T)

        # limit exponent so exp() doesn't overflow
        exponent = max(-10.0, min(10.0, exponent))

        # compute temperature-adjusted resistance
        R_temp = R * float(np.exp(exponent))

        # cap for stability (just a safety guard)
        return min(R_temp, 1.0)

    def f(self, y: np.ndarray, I: float) -> np.ndarray:
        """
        This function defines the ODE system: dy/dt = f(y, I).

        Real-world interpretation:
        - Current I causes Joule heating: Q_gen = I^2 * R_temp
        - Battery loses heat to air:    Q_loss = h*A*(T - T_amb)
        - SoC decreases with current draw: dSoC/dt = -|I|/Q_max
        - Resistance increases when hot and when SoC is low

        Returns:
          [dT/dt, dSoC/dt, dR/dt]
        """
        # Ensure current state is physically valid
        y = self._clamp_state(y)
        T, SoC, R = float(y[0]), float(y[1]), float(y[2])

        # temperature-dependent resistance
        R_temp = self._arrhenius_resistance(T, R)

        # heat generation (Joule heating) and cooling (convection)
        Q_gen = (abs(I) ** 2) * R_temp
        Q_loss = self.h * self.A * (T - self.T_amb)

        # ODE equations
        dT_dt = (Q_gen - Q_loss) / (self.m * self.c_p)   # temperature change rate
        dSoC_dt = -abs(I) / self.Q_max                   # SoC drop rate
        dR_dt = self.k_R * (T - self.T_ref) * (1.0 - SoC)  # resistance growth rate

        return np.array([dT_dt, dSoC_dt, dR_dt], dtype=float)

    def fixed_point_iteration(self, y_n: np.ndarray, dt: float, I: float,
                              tol: float = 1e-6, max_iter: int = 100):
        """
        Solve one implicit Euler time step using fixed-point iteration.

        Implicit Euler equation:
          y_{n+1} = y_n + dt * f(y_{n+1}, I)

        Fixed-point iteration:
          start with y_new = y_n
          repeat:
            y_new = y_n + dt * f(y_old, I)
          until ||y_new - y_old|| < tol

        Returns:
          y_{n+1}, number_of_iterations_used
        """
        # initial guess: "next state is same as current state"
        y_new = y_n.copy()

        for k in range(max_iter):
            y_old = y_new.copy()

            # fixed-point update (uses f evaluated at previous iterate)
            y_new = y_n + dt * self.f(y_old, I)

            # safety clamp to keep values physical
            y_new = self._clamp_state(y_new)

            # check if iteration converged
            error = float(np.linalg.norm(y_new - y_old))
            if error < tol:
                return y_new, (k + 1)

        # if not converged, return best found value anyway
        return y_new, max_iter

    def solve(self, t_span, y0, I_func, dt: float = 0.5, tol: float = 1e-6):
        """
        Time integration loop (march forward in time).

        Inputs:
          t_span = (start_time, end_time)
          y0     = initial state [T0, SoC0, R0]
          I_func = function I(t) giving current profile
          dt     = time step size
          tol    = fixed-point tolerance

        Output:
          dict with arrays for t, T, SoC, R + performance metrics
        """
        # build discrete time grid
        t0, t1 = float(t_span[0]), float(t_span[1])
        t = np.arange(t0, t1 + dt, dt, dtype=float)
        n_steps = len(t)

        # allocate solution array: y[time_index, state_index]
        y = np.zeros((n_steps, 3), dtype=float)

        # set initial condition (clamped)
        y[0] = self._clamp_state(np.array(y0, dtype=float))

        # performance tracking (how many iterations, how long time)
        total_iters = 0
        iter_counts = []
        start = time.time()

        # main time loop: compute y[i+1] from y[i]
        for i in range(n_steps - 1):
            # current for the next time step
            I = float(I_func(t[i + 1]))

            # implicit Euler step solved by fixed-point iteration
            y[i + 1], iters = self.fixed_point_iteration(y[i], dt, I, tol=tol)
            total_iters += iters
            iter_counts.append(iters)

            # stop early if battery is fully discharged
            if y[i + 1, 1] <= 0.0:
                y[i + 1:, 1] = 0.0
                break

        # compute performance summary
        cpu_time = time.time() - start
        avg_iters = total_iters / max(1, (n_steps - 1))

        # return results in a nice dictionary
        return {
            "t": t,
            "T": y[:, 0],
            "SoC": y[:, 1],
            "R": y[:, 2],
            "cpu_time": cpu_time,
            "total_iterations": total_iters,
            "avg_iterations": avg_iters,
            "min_iterations": int(min(iter_counts)) if iter_counts else 0,
            "max_iterations": int(max(iter_counts)) if iter_counts else 0,
        }


# ==============================================================================
# CURRENT PROFILE FUNCTIONS
# ==============================================================================
def current_profile_normal(t: float) -> float:
    """Scenario 1: normal charging current (constant 20A)."""
    return 20.0


def current_profile_fast(t: float) -> float:
    """Scenario 2: fast charging current (constant 50A)."""
    return 50.0


def current_profile_extreme(t: float) -> float:
    """
    Scenario 3: extreme pulsed current:
    - 0..500s: 70A
    - 500..1000s: 35A
    - 1000..end: 70A
    """
    if t < 500.0:
        return 70.0
    if t < 1000.0:
        return 35.0
    return 70.0


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def ensure_output_dir(dir_name: str) -> str:
    """
    Create folder for saving results (plots).
    If it already exists, do nothing.
    """
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_figure(fig, out_dir: str, filename: str):
    """
    Save a matplotlib figure to a PNG file in the output directory.
    """
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


# ==============================================================================
# MAIN EXECUTION - FIXED-POINT METHOD ONLY
# ==============================================================================
if __name__ == "__main__":
    # simple console header
    print("\n" + "=" * 80)
    print("BATTERY THERMAL RUNAWAY - FIXED-POINT ITERATION METHOD")
    print("=" * 80)

    # create output directory for plots
    out_dir = ensure_output_dir("battery_results_fixedpoint")

    # create model object (contains parameters + solver)
    model = BatteryModelFixedPoint()

    # initial condition: 25°C, full charge, 0.05 Ohm
    y0 = np.array([298.0, 1.0, 0.05], dtype=float)

    # simulate 30 minutes
    t_span = (0.0, 1800.0)

    # time step
    dt = 0.5

    # list of scenarios to run (name + current function)
    scenarios = [
        ("Normal Charging (2C)", current_profile_normal),
        ("Fast Charging (5C)", current_profile_fast),
        ("Extreme Fast Charging (Pulsed)", current_profile_extreme),
    ]

    results_all = {}

    # run all scenarios and print summary stats
    for name, I_func in scenarios:
        print(f"\n{name}")

        # solve ODE system with fixed-point implicit Euler
        r = model.solve(t_span, y0, I_func, dt=dt, tol=1e-6)
        results_all[name] = r

        # performance + key outputs (easy to compare between scenarios)
        print(f"  CPU Time:           {r['cpu_time']:.4f} s")
        print(f"  Total Iterations:   {r['total_iterations']}")
        print(f"  Avg Iter/Step:      {r['avg_iterations']:.2f}")
        print(f"  Min Iter/Step:      {r['min_iterations']}")
        print(f"  Max Iter/Step:      {r['max_iterations']}")
        print(f"  Max Temperature:    {np.max(r['T']) - 273.0:.2f} °C")
        print(f"  Final SoC:          {r['SoC'][-1]:.4f}")
        print(f"  Final Resistance:   {r['R'][-1] * 1000.0:.2f} mΩ")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    # Create a 2x2 figure to show main outputs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Battery Thermal Runaway Dynamics - Fixed-Point Iteration",
                 fontsize=14, fontweight="bold")

    colors = ["blue", "orange", "red"]

    # -------- Plot 1: Temperature vs time --------
    ax = axes[0, 0]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["T"] - 273.0, label=name, color=colors[idx], linewidth=2)

    # safety temperature line (example threshold)
    ax.axhline(y=60.0, color="red", linestyle="--", alpha=0.5, label="Safety Limit (60°C)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Evolution", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -------- Plot 2: SoC vs time --------
    ax = axes[0, 1]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["SoC"] * 100.0, label=name, color=colors[idx], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("State of Charge Depletion", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -------- Plot 3: Resistance vs time --------
    ax = axes[1, 0]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["t"], r["R"] * 1000.0, label=name, color=colors[idx], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Internal Resistance (mΩ)")
    ax.set_title("Internal Resistance Growth", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -------- Plot 4: Phase plot (SoC vs Temperature) --------
    # Shows relation between SoC and temperature, not just time.
    ax = axes[1, 1]
    for idx, (name, r) in enumerate(results_all.items()):
        ax.plot(r["SoC"] * 100.0, r["T"] - 273.0, label=name, color=colors[idx], linewidth=2)

        # mark initial point
        ax.plot(r["SoC"][0] * 100.0, r["T"][0] - 273.0, "o", color=colors[idx], markersize=8)

    ax.set_xlabel("State of Charge (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Phase Portrait (T vs SoC)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # layout and save plot
    plt.tight_layout()
    save_figure(fig, out_dir, "battery_fixedpoint_results.png")
    plt.show()

    print("\n" + "=" * 80)
    print("FIXED-POINT ITERATION METHOD COMPLETED SUCCESSFULLY")
    print("=" * 80)
