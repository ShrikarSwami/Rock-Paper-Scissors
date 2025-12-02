# Rock Paper Scissors CV

Webcam based Rock Paper Scissors where you can play against an AI opponent or share the camera with a friend for split screen local multiplayer. The game uses OpenCV and a custom hand detector so you can throw real gestures instead of pressing buttons.

## Features

- **Two ways to play**
  - **Versus AI**. Stand in front of the camera and play Rock Paper Scissors against an automated opponent. The AI chooses its move while the game reads your hand gesture.
  - **Two player mode**. The screen is split into a red tinted left side and a blue tinted right side. Each player has a lane to place their hand and the game tracks both gestures at the same time.

- **Best of three with win by two**
  - The match score is tracked for each side.
  - First to at least three points and leading by two wins the match.
  - Works the same in AI and two player mode.

- **Visual countdown and match flow**
  - When a valid hand or both hands are detected the game starts a three second countdown.
  - A large timer and optional beeps lead into the final “GO” moment.
  - The game then locks in both gestures, shows symbols and the result, then returns to waiting for the next round.

- **In game controls**
  - `p` to pause.
  - `m` to return to the main menu.
  - `q` to quit from anywhere.

- **Sound effects (optional)**
  - Short beeps during the countdown.
  - Celebration and reaction sounds on wins and losses when your sound files are available.

## Opponents

You can go up against two kinds of opponents.

- **Computer opponent**. In single player mode the right side of the scoreboard belongs to the AI. The AI picks rock, paper or scissors for each round while your hand gesture is read from the webcam feed.
- **Human opponent**. In multi player mode another person stands on the other side of the frame. One player uses the red tinted half of the screen, the other uses the blue tinted half, and both throws are evaluated at the same time.

## How it works

The core pieces of the project are:

- `main.py`  
  Opens the webcam with OpenCV, sets up the main loop, handles the menu, state transitions and keyboard controls. It also defines the single player and multi player flow, countdown timing and score handling.

- `hand_detextor.py`  
  Exposes a `HandDetector` class that looks at each frame and returns predictions. Each prediction includes a gesture label and a center point, which lets the game decide whether a hand counts for the left player, the right player, or the single player.

- `game_logic.py`  
  Contains game specific helpers:
  - `GESTURES` list for the valid moves (rock, paper, scissors).
  - `ScoreTracker` which knows round scores, match scores and when someone has won a best of three with win by two.
  - Functions like `choose_ai_move`, `decide_winner` and `move_to_symbol` which pick the AI move, decide who won the round and format moves for display.

- `SoundPlayer` inside `main.py`  
  A small helper around `simpleaudio` that loads wav files from the `sounds/` folder (for example `beep.wav`, `cheer.wav`, `boo.wav`, `yay.wav`). If audio is not available it safely falls back to a console beep so the game still runs.

Everything is drawn directly onto the camera frames using OpenCV, including overlays for text, countdown, scores and the red or blue tint for each side in two player mode.

## Requirements

- Python 3.10  
  MediaPipe and some of the vision related tooling work best here, and Python 3.12 plus often breaks hand tracking stacks.
- A working webcam.
- A reasonably recent version of `pip`.

Python libraries used:

- `opencv-python` for video capture and drawing.
- `mediapipe` or a similar backend inside `hand_detextor.py` to perform hand tracking.
- `simpleaudio` for sound playback (optional, but recommended for full experience).

## Setup

Clone the repository:

```bash
git clone https://github.com/ShrikarSwami/Rock-Paper-Scissors.git
cd Rock-Paper-Scissors
