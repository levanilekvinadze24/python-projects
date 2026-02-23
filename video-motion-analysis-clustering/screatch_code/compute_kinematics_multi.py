import csv
import math
import os
import matplotlib.pyplot as plt

# ===== PATHS =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)          # PythonProject
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_CSV = os.path.join(RESULTS_DIR, "pacman_multi_positions.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "pacman_multi_kinematics.csv")


def finite_diff(values: list[float], dt: float) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n

    deriv = [0.0] * n
    deriv[0] = (values[1] - values[0]) / dt
    deriv[-1] = (values[-1] - values[-2]) / dt

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
    print("🔎 Reading:", INPUT_CSV)
    if not os.path.exists(INPUT_CSV):
        print("❌ Cannot find positions CSV:", INPUT_CSV)
        return

    # object_id -> { frame: [], t: [], x: [], y: [] }
    data: dict[str, dict[str, list[float]]] = {}

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object_id"]
            frame = int(row["frame"])
            t_val = float(row["t_sec"])
            x = float(row["x"])
            y = float(row["y"])

            if obj not in data:
                data[obj] = {"frame": [], "t": [], "x": [], "y": []}

            data[obj]["frame"].append(frame)
            data[obj]["t"].append(t_val)
            data[obj]["x"].append(x)
            data[obj]["y"].append(y)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_rows: list[list] = []

    for obj, d in data.items():
        frames_raw = d["frame"]
        t_raw = d["t"]
        xs_raw = d["x"]
        ys_raw = d["y"]

        if len(t_raw) < 5:
            print(f"ℹ {obj}: not enough points, skipping")
            continue

        # --- 1) Remove duplicates within the same frame by averaging ---
        frame_map: dict[int, dict[str, float]] = {}

        for i in range(len(frames_raw)):
            f = frames_raw[i]
            t_val = t_raw[i]
            x_val = xs_raw[i]
            y_val = ys_raw[i]

            if f not in frame_map:
                frame_map[f] = {"t": t_val, "x": x_val, "y": y_val, "count": 1}
            else:
                frame_map[f]["t"] += t_val
                frame_map[f]["x"] += x_val
                frame_map[f]["y"] += y_val
                frame_map[f]["count"] += 1

        frames = sorted(frame_map.keys())
        t: list[float] = []
        xs: list[float] = []
        ys: list[float] = []

        for f in frames:
            rec = frame_map[f]
            c = rec["count"]
            t.append(rec["t"] / c)
            xs.append(rec["x"] / c)
            ys.append(rec["y"] / c)

        if len(t) < 5:
            print(f"ℹ {obj}: too few points after merging duplicates, skipping")
            continue

        dt = t[1] - t[0]
        if abs(dt) < 1e-9:
            print(f"⚠ {obj}: dt == 0 (cannot compute derivatives), skipping")
            continue

        vx = finite_diff(xs, dt)
        vy = finite_diff(ys, dt)
        v = [math.hypot(vx[i], vy[i]) for i in range(len(vx))]

        ax = finite_diff(vx, dt)
        ay = finite_diff(vy, dt)
        a = [math.hypot(ax[i], ay[i]) for i in range(len(ax))]

        jerk = finite_diff(a, dt)
        jounce = finite_diff(jerk, dt)

        v_s = moving_avg(v, window=5)
        a_s = moving_avg(a, window=5)
        jerk_s = moving_avg(jerk, window=5)
        jounce_s = moving_avg(jounce, window=5)

        for i in range(len(frames)):
            out_rows.append([
                obj,
                frames[i],
                t[i],
                xs[i],
                ys[i],
                v_s[i],
                a_s[i],
                jerk_s[i],
                jounce_s[i],
            ])

        # optional per-object speed plot
        plt.figure()
        plt.plot(t, v_s)
        plt.xlabel("t [s]")
        plt.ylabel("Speed [pixels/s]")
        plt.title(f"Speed over time: {obj}")
        plt.grid(True)
        out_png = os.path.join(RESULTS_DIR, f"speed_{obj}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"📊 speed plot saved: {out_png}")

    # combined kinematics CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "object_id", "frame", "t", "x", "y",
            "v", "a", "jerk", "jounce",
        ])
        writer.writerows(out_rows)

    print("✅ Multi-object kinematics table saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
