"""
Hand detection and gesture classification powered by MediaPipe.
"""

from dataclasses import dataclass
from math import acos, degrees
from typing import List, Tuple

import cv2

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - dependency hint
    raise RuntimeError(
        "mediapipe is required for hand tracking. Install with `pip install mediapipe`."
    ) from exc


@dataclass
class HandPrediction:
    gesture: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    handedness: str
    landmarks: List[Tuple[int, int]]


class HandDetector:
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7, tracking_confidence: float = 0.6):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame, draw: bool = False) -> List[HandPrediction]:
        """
        Detect hands in a frame and classify them into rock/paper/scissors/unknown.

        Returns a list of HandPrediction objects.
        """
        predictions: List[HandPrediction] = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return predictions

        h, w, _ = frame.shape
        handedness = result.multi_handedness or []

        for idx, lm_set in enumerate(result.multi_hand_landmarks):
            coords = [(int(pt.x * w), int(pt.y * h)) for pt in lm_set.landmark]
            label = handedness[idx].classification[0].label if idx < len(handedness) else "Unknown"
            gesture = self._classify(coords, label)
            bbox = self._bbox(coords, w, h)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            prediction = HandPrediction(
                gesture=gesture,
                center=center,
                bbox=bbox,
                handedness=label,
                landmarks=coords,
            )
            predictions.append(prediction)

            if draw:
                self.drawer.draw_landmarks(frame, lm_set, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{gesture}",
                    (bbox[0], max(20, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
        return predictions

    def _classify(self, coords: List[Tuple[int, int]], hand_label: str) -> str:
        """
        Classify a set of 21 landmarks into a gesture.

        Uses joint angles so side-on and palm-up views both work.
        """
        if len(coords) < 21:
            return "unknown"

        thumb = (0, 2, 4)  # wrist, thumb_mcp, thumb_tip
        fingers = [
            (5, 6, 8),  # index
            (9, 10, 12),  # middle
            (13, 14, 16),  # ring
            (17, 18, 20),  # pinky
        ]

        extended = []
        # Thumb: angle at MCP tells us if the thumb is out.
        thumb_angle = self._angle(coords[thumb[0]], coords[thumb[1]], coords[thumb[2]])
        extended.append(thumb_angle > 140)  # slightly looser for side views

        # Fingers: angle at PIP for bend/straight.
        for a, b, c in fingers:
            angle = self._angle(coords[a], coords[b], coords[c])
            extended.append(angle > 150)

        count = sum(extended)

        # Rock: fist (0 or 1 extended to allow tiny thumb noise).
        if count <= 1:
            return "rock"
        # Scissors: exactly index and middle extended.
        if count >= 2 and extended[1] and extended[2] and not (extended[3] and extended[4]):
            return "scissors"
        # Paper: all fingers open (accept 4-5 to be tolerant of partial thumb view).
        if count >= 4:
            return "paper"
        return "unknown"

    @staticmethod
    def _angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
        """Return the interior angle at p2 formed by p1-p2-p3."""
        ax, ay = p1[0] - p2[0], p1[1] - p2[1]
        bx, by = p3[0] - p2[0], p3[1] - p2[1]
        dot = ax * bx + ay * by
        mag_a = (ax * ax + ay * ay) ** 0.5
        mag_b = (bx * bx + by * by) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return degrees(acos(cos_theta))

    @staticmethod
    def _bbox(coords: List[Tuple[int, int]], width: int, height: int) -> Tuple[int, int, int, int]:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x1, x2 = max(0, min(xs)), min(width - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(height - 1, max(ys))
        return x1, y1, x2, y2

    def close(self) -> None:
        self.hands.close()
