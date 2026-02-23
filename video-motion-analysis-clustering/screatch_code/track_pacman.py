import cv2
import numpy as np
import csv
import os

VIDEO_PATH = "../videos/pacman.mp4"
OUTPUT_CSV = "../results/pacman_positions.csv"


def track_pacman() -> None:
    # make sure the results folder exists
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
    positions: list[tuple[int, int, int]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Pac-Man is yellow – move to HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow range in HSV (OpenCV: H in [0, 179])
        # This range may need a bit of tuning, but it is a good starting point
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # basic noise cleaning
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cx = cy = None

        if contours:
            h, w, _ = frame.shape
            frame_area = w * h

            # remove very small yellow dots (text, UI, etc.)
            min_area = frame_area * 0.0005   # 0.05% of the frame area
            max_area = frame_area * 0.2      # ignore if it is too large

            candidates = [
                c for c in contours
                if min_area < cv2.contourArea(c) < max_area
            ]

            if candidates:
                # take the largest yellow object – this should be Pac-Man
                c = max(candidates, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    positions.append((frame_idx, cx, cy))

                    # draw a red circle and a green bounding box on Pac-Man
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # visualisation
        cv2.imshow("Pac-Man tracking", frame)
        cv2.imshow("Yellow mask", mask)

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # time computation
    dt = 1.0 / fps

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_sec", "x", "y"])
        for frame, x, y in positions:
            t = frame * dt
            writer.writerow([frame, t, x, y])

    print(f"✅ Saved {len(positions)} points to file: {OUTPUT_CSV}")


if __name__ == "__main__":
    track_pacman()
