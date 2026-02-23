import csv
import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# ===== PATHS =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)           # PythonProject
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "pacman_multi_kinematics.csv")


def load_features():
    print("🔎 Looking for kinematics CSV:", INPUT_CSV)
    if not os.path.exists(INPUT_CSV):
        print("❌ Cannot find kinematics CSV:", INPUT_CSV)
        return None, None, None

    data: dict[str, list[dict]] = {}

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object_id"]
            if obj not in data:
                data[obj] = []
            data[obj].append(row)

    object_ids: list[str] = []
    features_pos: list[list[float]] = []
    features_dyn: list[list[float]] = []

    for obj, rows in data.items():
        xs = [float(r["x"]) for r in rows]
        ys = [float(r["y"]) for r in rows]
        vs = [float(r["v"]) for r in rows]
        acs = [float(r["a"]) for r in rows]
        jks = [float(r["jerk"]) for r in rows]
        jns = [float(r["jounce"]) for r in rows]

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))

        avg_v = float(np.mean(np.abs(vs)))
        max_v = float(np.max(np.abs(vs)))
        avg_a = float(np.mean(np.abs(acs)))
        avg_j = float(np.mean(np.abs(jks)))
        avg_jn = float(np.mean(np.abs(jns)))

        object_ids.append(obj)

        # position-based features
        features_pos.append([mean_x, mean_y])

        # derivative-based features
        features_dyn.append([avg_v, max_v, avg_a, avg_j, avg_jn])

    return object_ids, np.array(features_pos), np.array(features_dyn)


def main():
    obj_ids, X_pos, X_dyn = load_features()
    if obj_ids is None:
        return

    n_objects = len(obj_ids)
    if n_objects == 0:
        print("❌ No objects found in kinematics file.")
        return

    print("🎯 Objects:", obj_ids)
    n_clusters = min(3, n_objects)

    print("\n=== KMeans clustering (L2 norm) on POSITION only ===")
    kmeans_pos = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels_pos = kmeans_pos.fit_predict(X_pos)
    for obj, lab in zip(obj_ids, labels_pos):
        print(f"{obj:12s} -> cluster {lab}")

    print("\n=== KMeans clustering (L2 norm) on DERIVATIVE features ===")
    kmeans_dyn = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels_dyn = kmeans_dyn.fit_predict(X_dyn)
    for obj, lab in zip(obj_ids, labels_dyn):
        print(f"{obj:12s} -> cluster {lab}")

    print("\n=== Agglomerative clustering with L1 (Manhattan) on DERIVATIVE features ===")
    agg_l1 = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="manhattan",   # previously affinity="manhattan"
        linkage="average",
    )
    labels_dyn_l1 = agg_l1.fit_predict(X_dyn)
    for obj, lab in zip(obj_ids, labels_dyn_l1):
        print(f"{obj:12s} -> cluster {lab} (L1)")


if __name__ == "__main__":
    main()
