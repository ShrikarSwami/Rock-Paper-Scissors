import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import cv2

from game_logic import GESTURES, ScoreTracker, choose_ai_move, decide_winner
from hand_detextor import HandDetector

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
PFP_DIR = ASSETS_DIR / "pfps"
WINDOW_NAME = "Rock Paper Scissors"
COUNTDOWN_SECONDS = 3  # show 3-2-1 then GO
RESULT_HOLD_SECONDS = 5.0
AI_NAMES = ["Kandiddy", "Dhir", "JigglyDith", "JigglyPathi", "Skittles", "Pay Gorn", "Sigeon Pex", "Vis"]
SOUND_FILES = {
    "countdown": SOUNDS_DIR / "countdown.mp3",  # plays on 3-2-1 ticks
    "go": SOUNDS_DIR / "go.mp3",  # plays on GO
    "cheer": SOUNDS_DIR / "cheer.mp3",
    "boo": SOUNDS_DIR / "boo.mp3",
    "yay": SOUNDS_DIR / "yay.mp3",
}


class SoundPlayer:
    """
    Lightweight audio helper. Prefers pygame (mp3 support), falls back to simpleaudio for wav, otherwise console bell.
    """

    def __init__(self):
        self.loaded = {}
        self.backend = None
        self.pygame = None
        self.sa = None
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
        PFP_DIR.mkdir(parents=True, exist_ok=True)

        # Try pygame mixer for mp3 files.
        try:
            import pygame

            pygame.mixer.init()
            self.pygame = pygame
            self.backend = "pygame"
            for key, path in SOUND_FILES.items():
                if path.exists():
                    try:
                        self.loaded[key] = pygame.mixer.Sound(str(path))
                    except Exception:
                        pass
        except Exception:
            self.backend = None

        # Fallback: simpleaudio for wav (if user swaps files).
        if not self.loaded:
            try:
                import simpleaudio as sa

                self.sa = sa
                self.backend = "simpleaudio"
                for key, path in SOUND_FILES.items():
                    if path.exists() and path.suffix.lower() == ".wav":
                        try:
                            self.loaded[key] = sa.WaveObject.from_wave_file(str(path))
                        except Exception:
                            pass
            except Exception:
                self.backend = None

    def play(self, key: str) -> None:
        if key in self.loaded:
            if self.backend == "pygame":
                self.loaded[key].play()
            elif self.backend == "simpleaudio":
                self.loaded[key].play()
            return
        if key in ("countdown", "go"):
            print("\a", end="", flush=True)


def draw_text(frame, text: str, origin: Tuple[int, int], scale: float = 0.8, color=(255, 255, 255), bg=True) -> None:
    """Draw text with an optional dark background for readability."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    if bg:
        cv2.rectangle(frame, (x - 6, y - h - 6), (x + w + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_center_text(frame, text: str, y: int, scale: float = 1.0, color=(255, 255, 255), bg: bool = True) -> None:
    """Center text horizontally at a given y."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = frame.shape[1] // 2 - w // 2
    draw_text(frame, text, (x, y), scale=scale, color=color, bg=bg)


def draw_controls(frame) -> None:
    h, w, _ = frame.shape
    controls = ["p: pause", "m: menu", "q: quit"]
    for idx, label in enumerate(controls):
        draw_text(frame, label, (w - 180, 30 + idx * 30), scale=0.55)


def prompt_name(prompt: str, default: str) -> str:
    return default  # placeholder; not used anymore for GUI input


def text_input(prompt: str, default: str) -> str:
    """
    On-screen text input; type letters/numbers, backspace to delete, Enter to confirm, ESC to use default.
    """
    win = "Enter Text"
    cv2.namedWindow(win)
    text = ""
    while True:
        canvas = np.zeros((220, 640, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (630, 210), (80, 80, 80), 2)
        draw_text(canvas, prompt, (24, 60), scale=0.8, bg=False)
        draw_text(canvas, text or default, (24, 130), scale=1.0, bg=False)
        draw_text(canvas, "Enter=OK  ESC=skip", (24, 180), scale=0.6, bg=False)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # Enter
            cv2.destroyWindow(win)
            return text or default
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            return default
        if key == 8:  # Backspace
            text = text[:-1]
        elif 32 <= key <= 126:
            text += chr(key)


def center_square(frame):
    h, w, _ = frame.shape
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frame[y0 : y0 + size, x0 : x0 + size]


def capture_avatar(name: str, device_index: int = 0) -> Optional[np.ndarray]:
    """
    Capture a square avatar photo for the given name and return it in memory.
    Press 'c' to capture, 's' to skip, or 'q' to abort.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Could not open camera for avatar capture; skipping.")
        return None

    window = f"Avatar Capture - {name}"
    avatar = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; skipping avatar capture.")
            break

        display = frame.copy()
        h, w, _ = display.shape
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        cv2.rectangle(display, (x0, y0), (x0 + size, y0 + size), (0, 255, 0), 2)
        draw_text(display, "Align face, press 'c' to capture, 's' to skip, 'q' to cancel", (20, 30), scale=0.6)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            avatar = center_square(frame)
            break
        if key in (ord("s"), ord("q"), 27):  # s or q or ESC to skip
            break

    cap.release()
    cv2.destroyWindow(window)
    return avatar


def setup_single_game() -> Dict:
    name = text_input("Enter your name", "You")
    avatar = capture_avatar(name)
    return new_single_ctx(name, avatar)


def setup_multi_game() -> Dict:
    left = text_input("Enter left player name", "Player 1")
    left_avatar = capture_avatar(left)
    right = text_input("Enter right player name", "Player 2")
    right_avatar = capture_avatar(right)
    return new_multi_ctx(left, left_avatar, right, right_avatar)


def new_single_ctx(player_name: str, player_avatar: Optional[np.ndarray], ai_name: Optional[str] = None) -> Dict:
    ai = ai_name or random.choice(AI_NAMES)
    if player_avatar is None:
        player_avatar = placeholder_avatar()
    return {
        "score": ScoreTracker(),
        "phase": "waiting",  # waiting -> countdown -> result
        "countdown_start": None,
        "last_beep_second": None,
        "last_move": None,
        "result_time": None,
        "player_move": None,
        "ai_move": None,
        "round_outcome": None,
        "match_winner": None,
        "player_name": player_name,
        "ai_name": ai,
        "player_avatar": player_avatar,
        "ai_avatar": load_avatar(ai),
    }


def new_multi_ctx(left_name: str, left_avatar: Optional[np.ndarray], right_name: str, right_avatar: Optional[np.ndarray]) -> Dict:
    if left_avatar is None:
        left_avatar = placeholder_avatar()
    if right_avatar is None:
        right_avatar = placeholder_avatar()
    return {
        "score": ScoreTracker(),
        "phase": "waiting",
        "countdown_start": None,
        "last_beep_second": None,
        "last_left": None,
        "last_right": None,
        "result_time": None,
        "left_move": None,
        "right_move": None,
        "round_outcome": None,
        "match_winner": None,
        "left_name": left_name,
        "right_name": right_name,
        "left_avatar": left_avatar,
        "right_avatar": right_avatar,
    }


def render_menu(frame):
    overlay = frame.copy()
    draw_text(overlay, "Rock, Paper, Scissors", (40, 70), scale=1.1)
    draw_text(overlay, "1 - Play vs AI", (40, 120))
    draw_text(overlay, "2 - Two players", (40, 160))
    draw_text(overlay, "Press key to choose mode, or q to quit", (40, 210), scale=0.65)
    return overlay


def render_countdown(frame, start_time: float, sound: SoundPlayer, last_second: int):
    elapsed = time.time() - start_time
    remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))
    h, w, _ = frame.shape
    draw_text(frame, f"{remaining or 'GO!'}", (w // 2 - 50, h // 2), scale=2.5)

    if sound and remaining != last_second:
        sound.play("countdown" if remaining else "go")
        last_second = remaining
    return last_second


AVATAR_CACHE: Dict[str, Optional[np.ndarray]] = {}


def placeholder_avatar() -> np.ndarray:
    img = np.full((128, 128, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (127, 127), (200, 200, 200), 2)
    cv2.putText(img, "PFP", (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 2, cv2.LINE_AA)
    return img


def load_avatar(name: str):
    """Load persistent avatars for AI/placeholder from disk; user captures are in-memory."""
    if not name:
        name = "placeholder"
    if name in AVATAR_CACHE:
        return AVATAR_CACHE[name]
    for ext in (".png", ".jpg", ".jpeg"):
        path = PFP_DIR / f"{name}{ext}"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                AVATAR_CACHE[name] = img
                return img
    ph = AVATAR_CACHE.get("__placeholder__")
    if ph is None:
        ph = placeholder_avatar()
        AVATAR_CACHE["__placeholder__"] = ph
    AVATAR_CACHE[name] = ph
    return ph


def _rounded_mask(size: int, radius: int) -> np.ndarray:
    """Create a rounded-rectangle mask (uint8)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (size - radius, size), 255, -1)
    cv2.rectangle(mask, (0, radius), (size, size - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (size - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, size - radius), radius, 255, -1)
    cv2.circle(mask, (size - radius, size - radius), radius, 255, -1)
    return mask


def paste_avatar(card, avatar, top_left: Tuple[int, int]) -> None:
    if avatar is None:
        return
    size = 64
    radius = 12
    av = cv2.resize(avatar, (size, size))
    mask = _rounded_mask(size, radius)
    x, y = top_left
    h, w, _ = card.shape
    if y + size > h or x + size > w:
        return
    roi = card[y : y + size, x : x + size]
    mask_f = mask[:, :, None] / 255.0
    blended = (av * mask_f + roi * (1 - mask_f)).astype(np.uint8)
    card[y : y + size, x : x + size] = blended


def draw_move_cards(
    frame,
    left: Tuple[str, str, Optional[np.ndarray]],
    right: Tuple[str, str, Optional[np.ndarray]],
) -> None:
    """Render large cards showing moves so it's obvious who threw what."""
    h, w, _ = frame.shape
    card_w, card_h = 240, 180
    gap = 40
    left_x = w // 2 - card_w - gap // 2
    right_x = w // 2 + gap // 2
    y = h // 2 - card_h // 2

    for (label, move, avatar), x in ((left, left_x), (right, right_x)):
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (200, 200, 200), 2)
        mv = move or ""
        symbol = mv.upper() if mv in GESTURES else ""
        move_label = mv if mv in GESTURES else "unknown"
        draw_text(frame, label, (x + 12, y + 28), scale=0.8, bg=False)
        draw_text(frame, symbol, (x + 20, y + 110), scale=1.6, bg=False)
        draw_text(frame, move_label, (x + 12, y + card_h - 20), scale=0.7, bg=False)
        paste_avatar(frame[y : y + card_h, x : x + card_w], avatar, (card_w - 76, 12))


def draw_corner_profiles(frame, left: Tuple[str, np.ndarray], right: Tuple[str, np.ndarray]) -> None:
    """Always display names + avatars in bottom corners."""
    h, w, _ = frame.shape
    size = 64
    pad = 12
    # Left
    lx, ly = pad, h - size - pad
    cv2.rectangle(frame, (lx - 4, ly - 4), (lx + size + 140, ly + size + 8), (0, 0, 0), -1)
    cv2.rectangle(frame, (lx - 4, ly - 4), (lx + size + 140, ly + size + 8), (120, 120, 120), 1)
    paste_avatar(frame[ly : ly + size, lx : lx + size], left[1], (0, 0))
    draw_text(frame, left[0], (lx + size + 10, ly + size - 18), scale=0.7, bg=False)
    # Right
    rx, ry = w - size - pad - 140, h - size - pad
    cv2.rectangle(frame, (rx - 4, ry - 4), (rx + size + 140, ry + size + 8), (0, 0, 0), -1)
    cv2.rectangle(frame, (rx - 4, ry - 4), (rx + size + 140, ry + size + 8), (120, 120, 120), 1)
    paste_avatar(frame[ry : ry + size, rx : rx + size], right[1], (0, 0))
    draw_text(frame, right[0], (rx + size + 10, ry + size - 18), scale=0.7, bg=False)


def play_round_sound(sound: SoundPlayer, mode: str, result: str) -> None:
    if not sound:
        return
    if mode == "single":
        if result == "p1":
            sound.play("cheer")
        elif result == "p2":
            sound.play("boo")
    elif mode == "multi":
        if result in ("p1", "p2"):
            sound.play("yay")


def single_player(frame, key, detector: HandDetector, ctx: Dict, sound: SoundPlayer):
    if key == ord("m"):
        return frame, None, "menu"
    if key == ord("r") and ctx.get("match_winner"):
        return frame, new_single_ctx(ctx["player_name"], ctx["player_avatar"]), "single"

    predictions = detector.detect(frame, draw=True)
    for p in predictions:
        if p.gesture in GESTURES:
            ctx["last_move"] = p.gesture
            break
    draw_controls(frame)
    draw_text(frame, ctx["score"].as_text((ctx["player_name"], ctx["ai_name"])), (40, 40))
    draw_corner_profiles(frame, (ctx["player_name"], ctx["player_avatar"]), (ctx["ai_name"], ctx["ai_avatar"]))

    if ctx["match_winner"]:
        label = f"{ctx['player_name']} wins the match!" if ctx["match_winner"] == "p1" else f"{ctx['ai_name']} wins the match!"
        color = (0, 255, 0) if ctx["match_winner"] == "p1" else (0, 0, 255)
        draw_center_text(frame, label, 70, scale=1.3, color=color)
        draw_center_text(frame, "r: rematch  m: menu  q: quit", 120, scale=0.8, color=(255, 255, 255))
        return frame, ctx, "single"

    if ctx["phase"] == "waiting":
        draw_text(frame, f"{ctx['player_name']}: show your hand to start", (40, 90), scale=0.7)
        if any(p.gesture in GESTURES for p in predictions):
            ctx["phase"] = "countdown"
            ctx["countdown_start"] = time.time()
            ctx["last_beep_second"] = None
            ctx["last_move"] = None
    elif ctx["phase"] == "countdown":
        ctx["last_beep_second"] = render_countdown(frame, ctx["countdown_start"], sound, ctx["last_beep_second"])
        if time.time() - ctx["countdown_start"] >= COUNTDOWN_SECONDS:
            final_preds = detector.detect(frame)
            player_move = next((p.gesture for p in final_preds if p.gesture in GESTURES), ctx["last_move"] or "unknown")
            # Special AI behaviors
            if ctx["ai_name"] == "Sigeon Pex" and player_move in GESTURES:
                ai_move = {"rock": "paper", "paper": "scissors", "scissors": "rock"}[player_move]
            elif ctx["ai_name"] == "Vis" and player_move in GESTURES:
                ai_move = {"rock": "scissors", "paper": "rock", "scissors": "paper"}[player_move]
            else:
                ai_move = choose_ai_move()
            result = decide_winner(player_move, ai_move)
            if result in ("p1", "p2"):
                ctx["score"].apply_round(result)
                play_round_sound(sound, "single", result)
            ctx.update(
                {
                    "player_move": player_move,
                    "ai_move": ai_move,
                    "round_outcome": result,
                    "phase": "result",
                    "result_time": time.time(),
                    "match_winner": ctx["score"].winner(),
                }
            )
    elif ctx["phase"] == "result":
        draw_move_cards(
            frame,
            (ctx["player_name"], ctx["player_move"] or "unknown", ctx["player_avatar"]),
            (ctx["ai_name"], ctx["ai_move"] or "unknown", ctx["ai_avatar"]),
        )
        outcome = ""
        color = (255, 255, 255)
        if ctx["round_outcome"] == "tie":
            outcome = "Tie - go again"
            color = (255, 255, 255)
        elif ctx["round_outcome"] == "invalid":
            outcome = "Could not read your hand, try again"
            color = (0, 0, 255)
        elif ctx["round_outcome"] == "p1":
            outcome = "You win the round"
            color = (0, 200, 0)
        else:
            outcome = f"{ctx['ai_name']} wins the round"
            color = (0, 0, 255)
        draw_center_text(frame, outcome, 70, scale=1.1, color=color)

        if ctx["match_winner"]:
            label = f"{ctx['player_name']} wins the match!" if ctx["match_winner"] == "p1" else f"{ctx['ai_name']} wins the match!"
            color = (0, 255, 0) if ctx["match_winner"] == "p1" else (0, 0, 255)
            draw_center_text(frame, label, 110, scale=1.0, color=color)
        elif time.time() - ctx["result_time"] > RESULT_HOLD_SECONDS:
            ctx["phase"] = "waiting"
            ctx["last_beep_second"] = None
            ctx["last_move"] = None

    return frame, ctx, "single"


def multi_player(frame, key, detector: HandDetector, ctx: Dict, sound: SoundPlayer):
    if key == ord("m"):
        return frame, None, "menu"
    if key == ord("r") and ctx.get("match_winner"):
        return frame, new_multi_ctx(ctx["left_name"], ctx["left_avatar"], ctx["right_name"], ctx["right_avatar"]), "multi"

    h, w, _ = frame.shape
    mid_x = w // 2

    det_input = frame.copy()
    clean_input = det_input.copy()
    predictions = detector.detect(det_input, draw=True)
    # Start display from detection frame so landmarks/labels stay visible.
    frame[:, :, :] = det_input

    # Tinted split to help players position themselves (applied after detection).
    left_overlay = frame.copy()
    right_overlay = frame.copy()
    cv2.rectangle(left_overlay, (0, 0), (mid_x, h), (255, 0, 0), -1)
    cv2.rectangle(right_overlay, (mid_x, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(left_overlay, 0.08, frame, 0.92, 0, frame)
    cv2.addWeighted(right_overlay, 0.08, frame, 0.92, 0, frame)
    cv2.line(frame, (mid_x, 0), (mid_x, h), (200, 200, 200), 1)

    draw_controls(frame)
    draw_text(frame, ctx["score"].as_text((ctx["left_name"], ctx["right_name"])), (40, 40))
    draw_corner_profiles(frame, (ctx["left_name"], ctx["left_avatar"]), (ctx["right_name"], ctx["right_avatar"]))

    if ctx["match_winner"]:
        label = f"{ctx['left_name']} wins the match" if ctx["match_winner"] == "p1" else f"{ctx['right_name']} wins the match"
        color = (0, 255, 0) if ctx["match_winner"] == "p1" else (0, 0, 255)
        draw_center_text(frame, label, 70, scale=1.3, color=color)
        draw_center_text(frame, "r: rematch  m: menu  q: quit", 120, scale=0.8, color=(255, 255, 255))
        return frame, ctx, "multi"

    left = None
    right = None
    for pred in predictions:
        if pred.gesture not in GESTURES:
            continue
        if pred.center[0] < mid_x and left is None:
            left = pred
            ctx["last_left"] = pred.gesture
        elif pred.center[0] >= mid_x and right is None:
            right = pred
            ctx["last_right"] = pred.gesture

    if ctx["phase"] == "waiting":
        draw_center_text(frame, "Get ready: 3-2-1-GO", 90, scale=0.9)
        if ctx["countdown_start"] is None:
            ctx["countdown_start"] = time.time()
            ctx["phase"] = "countdown"
            ctx["last_beep_second"] = None
            ctx["last_left"] = None
            ctx["last_right"] = None
    elif ctx["phase"] == "countdown":
        ctx["last_beep_second"] = render_countdown(frame, ctx["countdown_start"], sound, ctx["last_beep_second"])
        if time.time() - ctx["countdown_start"] >= COUNTDOWN_SECONDS:
            final_preds = detector.detect(clean_input)
            left_move = next(
                (p.gesture for p in final_preds if p.center[0] < mid_x and p.gesture in GESTURES),
                ctx["last_left"] or "unknown",
            )
            right_move = next(
                (p.gesture for p in final_preds if p.center[0] >= mid_x and p.gesture in GESTURES),
                ctx["last_right"] or "unknown",
            )
            result = decide_winner(left_move, right_move)
            if result in ("p1", "p2"):
                ctx["score"].apply_round(result)
                play_round_sound(sound, "multi", result)
            ctx.update(
                {
                    "left_move": left_move,
                    "right_move": right_move,
                    "round_outcome": result,
                    "phase": "result",
                    "result_time": time.time(),
                    "match_winner": ctx["score"].winner(),
                    "countdown_start": None,
                }
            )
    elif ctx["phase"] == "result":
        draw_move_cards(
            frame,
            (ctx["left_name"], ctx["left_move"] or "unknown", ctx["left_avatar"]),
            (ctx["right_name"], ctx["right_move"] or "unknown", ctx["right_avatar"]),
        )
        outcome = ""
        color = (255, 255, 255)
        if ctx["round_outcome"] == "tie":
            outcome = "Tie - go again"
            color = (255, 255, 255)
        elif ctx["round_outcome"] == "invalid":
            outcome = "Could not read a hand, try again"
            color = (0, 0, 255)
        elif ctx["round_outcome"] == "p1":
            outcome = f"{ctx['left_name']} wins the round"
            color = (0, 200, 0)
        else:
            outcome = f"{ctx['right_name']} wins the round"
            color = (0, 0, 255)
        draw_center_text(frame, outcome, 70, scale=1.1, color=color)

        if ctx["match_winner"]:
            label = f"{ctx['left_name']} wins the match" if ctx["match_winner"] == "p1" else f"{ctx['right_name']} wins the match"
            color = (0, 255, 0) if ctx["match_winner"] == "p1" else (0, 0, 255)
            draw_center_text(frame, label, 110, scale=1.0, color=color)
        elif time.time() - ctx["result_time"] > RESULT_HOLD_SECONDS:
            ctx["phase"] = "waiting"
            ctx["last_beep_second"] = None
            ctx["last_left"] = None
            ctx["last_right"] = None

    return frame, ctx, "multi"


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector()
    sound = SoundPlayer()
    state = "menu"
    single_ctx = None
    multi_ctx = None
    paused = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed, exiting.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused

        if paused:
            draw_text(frame, "Paused - press p to resume", (40, 40))
            cv2.imshow(WINDOW_NAME, frame)
            continue

        if state == "menu":
            frame = render_menu(frame)
            if key == ord("1"):
                single_ctx = setup_single_game()
                state = "single"
            elif key == ord("2"):
                multi_ctx = setup_multi_game()
                state = "multi"
        elif state == "single":
            if single_ctx is None:
                single_ctx = setup_single_game()
            frame, single_ctx, state = single_player(frame, key, detector, single_ctx, sound)
        elif state == "multi":
            if multi_ctx is None:
                multi_ctx = setup_multi_game()
            frame, multi_ctx, state = multi_player(frame, key, detector, multi_ctx, sound)
        if state == "quit":
            break

        cv2.imshow(WINDOW_NAME, frame)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
