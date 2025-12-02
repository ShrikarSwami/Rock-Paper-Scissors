import time
from pathlib import Path
from typing import Dict, Tuple

import cv2

from game_logic import GESTURES, ScoreTracker, choose_ai_move, decide_winner, move_to_symbol
from hand_detextor import HandDetector

BASE_DIR = Path(__file__).resolve().parent
SOUNDS_DIR = BASE_DIR / "assets" / "sounds"
WINDOW_NAME = "Rock Paper Scissors"
COUNTDOWN_SECONDS = 4  # slower so players can see AI choice reveal
RESULT_HOLD_SECONDS = 5.0
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
        SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

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


def draw_controls(frame) -> None:
    h, w, _ = frame.shape
    controls = ["p: pause", "m: menu", "q: quit"]
    for idx, label in enumerate(controls):
        draw_text(frame, label, (w - 180, 30 + idx * 30), scale=0.55)


def new_single_ctx() -> Dict:
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
    }


def new_multi_ctx() -> Dict:
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


def draw_move_cards(frame, left: Tuple[str, str], right: Tuple[str, str]) -> None:
    """Render large cards showing moves so it's obvious who threw what."""
    h, w, _ = frame.shape
    card_w, card_h = 240, 180
    gap = 40
    left_x = w // 2 - card_w - gap // 2
    right_x = w // 2 + gap // 2
    y = h // 2 - card_h // 2

    for (label, move), x in ((left, left_x), (right, right_x)):
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (200, 200, 200), 2)
        symbol = move_to_symbol(move)
        draw_text(frame, label, (x + 12, y + 28), scale=0.8, bg=False)
        draw_text(frame, symbol, (x + 20, y + 110), scale=2.5, bg=False)
        draw_text(frame, move, (x + 12, y + card_h - 20), scale=0.7, bg=False)


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
        return frame, new_single_ctx(), "menu"

    predictions = detector.detect(frame, draw=True)
    for p in predictions:
        if p.gesture in GESTURES:
            ctx["last_move"] = p.gesture
            break
    draw_controls(frame)
    draw_text(frame, ctx["score"].as_text(("You", "AI")), (40, 40))

    if ctx["match_winner"]:
        label = "You win the match!" if ctx["match_winner"] == "p1" else "AI wins the match!"
        draw_text(frame, label, (40, 90))
        draw_text(frame, "Press space to play again or m for menu", (40, 130), scale=0.65)
        if key == ord(" "):
            ctx = new_single_ctx()
        return frame, ctx, "single"

    if ctx["phase"] == "waiting":
        draw_text(frame, "Show your hand to start (rock/paper/scissors)", (40, 90), scale=0.7)
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
        draw_move_cards(frame, ("You", ctx["player_move"] or "unknown"), ("AI", ctx["ai_move"] or "unknown"))
        if ctx["round_outcome"] == "tie":
            outcome = "Tie - go again"
        elif ctx["round_outcome"] == "invalid":
            outcome = "Could not read your hand, try again"
        elif ctx["round_outcome"] == "p1":
            outcome = "You win the round"
        else:
            outcome = "AI wins the round"
        draw_text(frame, outcome, (40, 90), scale=0.9)

        if ctx["match_winner"]:
            # Keep showing winner message until player restarts.
            draw_text(frame, "Match point reached", (40, 170), scale=0.65)
        elif time.time() - ctx["result_time"] > RESULT_HOLD_SECONDS:
            ctx["phase"] = "waiting"
            ctx["last_beep_second"] = None
            ctx["last_move"] = None

    return frame, ctx, "single"


def multi_player(frame, key, detector: HandDetector, ctx: Dict, sound: SoundPlayer):
    if key == ord("m"):
        return frame, new_multi_ctx(), "menu"

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
    draw_text(frame, ctx["score"].as_text(("Left", "Right")), (40, 40))

    if ctx["match_winner"]:
        label = "Left player wins the match" if ctx["match_winner"] == "p1" else "Right player wins the match"
        draw_text(frame, label, (40, 90))
        draw_text(frame, "Press space to play again or m for menu", (40, 130), scale=0.65)
        if key == ord(" "):
            ctx = new_multi_ctx()
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
        draw_text(frame, "Both players: place a hand inside your half", (40, 90), scale=0.7)
        if left and right:
            ctx["phase"] = "countdown"
            ctx["countdown_start"] = time.time()
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
                }
            )
    elif ctx["phase"] == "result":
        draw_move_cards(frame, ("Left", ctx["left_move"] or "unknown"), ("Right", ctx["right_move"] or "unknown"))
        if ctx["round_outcome"] == "tie":
            outcome = "Tie - go again"
        elif ctx["round_outcome"] == "invalid":
            outcome = "Could not read a hand, try again"
        elif ctx["round_outcome"] == "p1":
            outcome = "Left player wins the round"
        else:
            outcome = "Right player wins the round"
        draw_text(frame, outcome, (40, 90), scale=0.9)

        if ctx["match_winner"]:
            draw_text(frame, "Match point reached", (40, 170), scale=0.65)
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
    single_ctx = new_single_ctx()
    multi_ctx = new_multi_ctx()
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
                single_ctx = new_single_ctx()
                state = "single"
            elif key == ord("2"):
                multi_ctx = new_multi_ctx()
                state = "multi"
        elif state == "single":
            frame, single_ctx, state = single_player(frame, key, detector, single_ctx, sound)
        elif state == "multi":
            frame, multi_ctx, state = multi_player(frame, key, detector, multi_ctx, sound)

        cv2.imshow(WINDOW_NAME, frame)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
