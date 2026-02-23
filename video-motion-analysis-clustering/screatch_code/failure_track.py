import cv2
import os

# ===== PATH =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "failure_video.mp4")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video:", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Try to detect a bright yellow or red object — generic range
        lower = (15, 80, 80)
        upper = (40, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)

        # Simple clean-up
        mask = cv2.medianBlur(mask, 7)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 300:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Failure tracking", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
