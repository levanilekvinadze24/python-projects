import cv2
import numpy as np
import csv
import os

VIDEO_PATH = "../videos/pacman.mp4"
OUTPUT_CSV = "../results/pacman_multi_positions.csv"


def track_pacman_multi() -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print("🎥 FPS =", fps)

    frame_idx = 0
    rows: list[list] = []  # object_id, frame, t_sec, x, y

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w, _ = frame.shape
        frame_area = w * h
        t_sec = frame_idx / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Color ranges (extended) ---

        # Pac-Man – yellow
        yellow_low = np.array([15, 100, 100])
        yellow_high = np.array([40, 255, 255])

        # Red ghosts – two ranges
        red_low1 = np.array([0, 100, 80])
        red_high1 = np.array([10, 255, 255])
        red_low2 = np.array([170, 100, 80])
        red_high2 = np.array([179, 255, 255])

        # Cyan ghost
        cyan_low = np.array([80, 80, 80])
        cyan_high = np.array([100, 255, 255])

        # Magenta / pink ghost
        mag_low = np.array([130, 60, 80])
        mag_high = np.array([170, 255, 255])

        color_ranges = [
            ("pacman", yellow_low, yellow_high, (0, 255, 255)),   # yellow
            ("ghost_red", red_low1, red_high1, (0, 0, 255)),      # red 1
            ("ghost_red", red_low2, red_high2, (0, 0, 255)),      # red 2
            ("ghost_cyan", cyan_low, cyan_high, (255, 255, 0)),   # cyan
            ("ghost_mag", mag_low, mag_high, (255, 0, 255)),      # magenta/pink
        ]

        # area filters
        min_area = frame_area * 0.0005   # filter out very small dots
        max_area = frame_area * 0.2      # ignore very large blobs

        found_this_frame = []

        for obj_id, low, high, draw_color in color_ranges:
            mask = cv2.inRange(hsv, low, high)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                continue

            candidates = [
                c for c in contours
                if min_area < cv2.contourArea(c) < max_area
            ]
            if not candidates:
                continue

            # take the largest object for this color
            c = max(candidates, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            rows.append([obj_id, frame_idx, t_sec, cx, cy])
            found_this_frame.append(obj_id)

            # draw Pac-Man slightly differently so he is easy to see
            if obj_id == "pacman":
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
            else:
                x, y, w_box, h_box = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), draw_color, 2)
                cv2.putText(
                    frame,
                    obj_id,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    draw_color,
                    1,
                    cv2.LINE_AA,
                )

        # small debug text
        if found_this_frame:
            text = " | ".join(found_this_frame)
            cv2.putText(
                frame,
                text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Pac-Man multi-object tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "frame", "t_sec", "x", "y"])
        writer.writerows(rows)

    print(f"✅ Multi-object data saved to: {OUTPUT_CSV}")
    print("Total rows written:", len(rows))


if __name__ == "__main__":
    track_pacman_multi()
