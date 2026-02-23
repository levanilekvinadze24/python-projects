import csv
import math
import os
import matplotlib.pyplot as plt

INPUT_CSV = "../results/pacman_positions.csv"
OUTPUT_CSV = "../results/pacman_kinematics.csv"


def finite_diff(values: list[float], dt: float) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n

    deriv = [0.0] * n

    # forward
    deriv[0] = (values[1] - values[0]) / dt
    # backward
    deriv[-1] = (values[-1] - values[-2]) / dt

    # central
    for i in range(1, n - 1):
        deriv[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)

    return deriv


def moving_avg(values: list[float], window: int = 5) -> list[float]:
    n = len(values)
    if n == 0 or window <= 1:
        return values[:]

    half = window // 2
    smoothed: list[float] = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        segment = values[start:end]
        smoothed.append(sum(segment) / len(segment))

    return smoothed


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        print("❌ Cannot find positions CSV:", INPUT_CSV)
        return

    frames: list[int] = []
    t: list[float] = []
    xs: list[float] = []
    ys: list[float] = []

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            t.append(float(row["t_sec"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))

    if len(t) < 5:
        print("ℹ Not enough points – insufficient for kinematics")
        return

    dt = t[1] - t[0]

    # velocity (vector)
    vx = finite_diff(xs, dt)
    vy = finite_diff(ys, dt)
    v = [math.hypot(vx[i], vy[i]) for i in range(len(vx))]

    # acceleration
    ax = finite_diff(vx, dt)
    ay = finite_diff(vy, dt)
    a = [math.hypot(ax[i], ay[i]) for i in range(len(ax))]

    # jerk
    jerk = finite_diff(a, dt)

    # jounce
    jounce = finite_diff(jerk, dt)

    # smoothing
    v_s = moving_avg(v, window=5)
    a_s = moving_avg(a, window=5)
    jerk_s = moving_avg(jerk, window=5)
    jounce_s = moving_avg(jounce, window=5)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t", "x", "y", "v", "a", "jerk", "jounce"])
        for i in range(len(frames)):
            writer.writerow([
                frames[i],
                t[i],
                xs[i],
                ys[i],
                v_s[i],
                a_s[i],
                jerk_s[i],
                jounce_s[i],
            ])

    print("✅ Kinematics table saved to:", OUTPUT_CSV)

    # plots
    plt.figure()
    plt.plot(t, v_s)
    plt.xlabel("t [s]")
    plt.ylabel("Speed [pixels/s]")
    plt.title("Pac-Man speed over time")
    plt.grid(True)
    plt.savefig("../results/pacman_speed.png", dpi=200)

    plt.figure()
    plt.plot(t, a_s)
    plt.xlabel("t [s]")
    plt.ylabel("Acceleration [pixels/s^2]")
    plt.title("Pac-Man acceleration over time")
    plt.grid(True)
    plt.savefig("../results/pacman_acceleration.png", dpi=200)

    plt.figure()
    plt.plot(t, jerk_s)
    plt.xlabel("t [s]")
    plt.ylabel("Jerk")
    plt.title("Pac-Man jerk over time")
    plt.grid(True)
    plt.savefig("../results/pacman_jerk.png", dpi=200)

    plt.figure()
    plt.plot(t, jounce_s)
    plt.xlabel("t [s]")
    plt.ylabel("Jounce")
    plt.title("Pac-Man jounce over time")
    plt.grid(True)
    plt.savefig("../results/pacman_jounce.png", dpi=200)

    print("📊 Plots saved in ../results/ directory")


if __name__ == "__main__":
    main()
