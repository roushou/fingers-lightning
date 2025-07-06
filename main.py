import cv2
import mediapipe as mp
import numpy as np
import random

FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
WRIST_INDEX = 0
MAX_HANDS = 2

def draw_lightning(
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 255),
    segments: int = 10,
):
    """Draw a lightning effect between two points."""
    frame_height, frame_width = img.shape[:2]
    points = [start]

    dx = (end[0] - start[0]) / segments
    dy = (end[1] - start[1]) / segments

    for i in range(1, segments):
        x = start[0] + dx * i + random.randint(-20, 20)
        y = start[1] + dy * i + random.randint(-20, 20)
        x = max(0, min(frame_width - 1, int(x)))
        y = max(0, min(frame_height - 1, int(y)))
        points.append((x, y))

    points.append(end)
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 2)

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    video_capture = cv2.VideoCapture(0)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            fingertip_positions = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get wrist position for fingertip-to-wrist lightning
                wrist = hand_landmarks.landmark[WRIST_INDEX]
                wrist_pos = (int(wrist.x * frame_width), int(wrist.y * frame_height))

                # Store fingertip positions for this hand
                hand_tips = {}
                for tip_index in FINGERTIP_INDICES:
                    fingertip = hand_landmarks.landmark[tip_index]
                    fingertip_pos = (int(fingertip.x * frame_width), int(fingertip.y * frame_height))
                    hand_tips[tip_index] = fingertip_pos

                    # Draw a circle at the fingertip
                    cv2.circle(frame, fingertip_pos, 10, (0, 0, 255), -1)

                    # Draw lightning from fingertip to wrist
                    draw_lightning(frame, fingertip_pos, wrist_pos)

                fingertip_positions.append(hand_tips)

            # If two hands are detected, draw lightning between corresponding fingertips
            if len(fingertip_positions) == 2:
                hand1_tips = fingertip_positions[0]
                hand2_tips = fingertip_positions[1]
                for tip_index in FINGERTIP_INDICES:
                    start_pos = hand1_tips[tip_index]
                    end_pos = hand2_tips[tip_index]
                    draw_lightning(frame, start_pos, end_pos)

        cv2.imshow("Finger Lightning", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
