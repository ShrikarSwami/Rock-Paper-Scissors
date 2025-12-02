# Rock Paper Scissors CV

A webcam based Rock Paper Scissors game built with OpenCV and real-time hand tracking.  
Play against an AI opponent or enter a split-screen mode for two players.  
The game detects real hand gestures and manages countdowns, scoring, and full match flow.

---

## Features

### 🎮 Game Modes

#### **Single Player (vs AI)**
- Detects your hand gesture from the webcam.
- AI selects its move each round.
- Displays results, handles scoring, and determines match winners.

#### **Two Player Mode**
- Screen is split vertically: left (red) for Player 1, right (blue) for Player 2.
- Detects both players' gestures simultaneously.
- Full competitive match flow.

---

### 🧠 Gameplay System

- Real time gesture detection using a custom MediaPipe-based detector.
- Three second countdown with on-screen timer and optional sound cues.
- Round result overlay including symbols and win/tie messages.
- Best of three with win by two scoring system.
- Match winner state with prompts to restart or return to menu.

---

### 🔊 Sound Effects

If `simpleaudio` is installed and sound files exist under `sounds/`, the game plays:
- Countdown beeps  
- “Go” sound  
- Win/loss reactions  

Missing audio files do not prevent the game from running.

---

## Opponents

### **🤖 AI Opponent**
- AI randomly selects Rock, Paper, or Scissors.
- Your gesture is read from the webcam and evaluated against the AI.

### **🧍🧍 Two Human Players**
- Two players stand on opposite halves of the camera frame.
- Gestures are read from both sides at the same time.

---

## File Overview

### `main.py`
- Controls webcam loop, UI rendering, countdowns, and game state transitions.
- Manages scoring, match endings, pause/menu behavior, and gesture interpretation.

### `game_logic.py`
- Defines valid gestures.
- Contains round result logic.
- AI move selection.
- Score tracking with “win by two” match end conditions.
- Helper functions for gesture symbols and formatting.

### `hand_detextor.py`
- Core hand detection logic.
- Converts landmarks or predictions into gesture labels.
- Returns predictions with gesture name and center coordinates.

### `sounds/`
Optional audio files:
- `beep.wav`
- `cheer.wav`
- `boo.wav`
- `yay.wav`

---

## Requirements

### Python Version
- **Python 3.10 recommended**
- Other versions may break MediaPipe compatibility.

### Dependencies
Install using:

```bash
pip install -r requirements.txt
Dependencies include:

opencv-python

mediapipe

simpleaudio (optional)

Setup
1. Clone the repository
bash
Copy code
git clone https://github.com/ShrikarSwami/Rock-Paper-Scissors.git
cd Rock-Paper-Scissors
2. Set Python version (if using pyenv)
bash
Copy code
pyenv local 3.10.12
3. Create and activate virtual environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate
4. Install dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
pip install simpleaudio
Running the Game
bash
Copy code
python main.py
Controls
1 — Start single player mode

2 — Start two player mode

p — Pause

m — Return to menu

q — Quit

Project Structure
arduino
Copy code
Rock-Paper-Scissors/
├── main.py
├── game_logic.py
├── hand_detextor.py
├── sounds/
│   ├── beep.wav
│   ├── cheer.wav
│   ├── boo.wav
│   └── yay.wav
├── requirements.txt
└── README.md
Future Ideas
Difficulty settings for AI

Larger and animated gesture icons

Match analytics/history

Gesture calibration mode

Additional gesture sets (e.g., Lizard Spock)

Animated overlays and polish

yaml
Copy code

---

If you want a **Markdown wiki homepage**, **installation guide**, **developer guide**, or **character wiki in the same style**, tell me and I’ll generate it the same way.





You said:
no you're doing it wrong

make it so its all in like the code way and not any prompt




ChatGPT said:
Understood.
Here is the README only as a Markdown code block, with no explanation before or after.

markdown
Copy code
# Rock Paper Scissors CV

A webcam based Rock Paper Scissors game built with OpenCV and real-time hand tracking.  
Play against an AI opponent or enter a split-screen mode for two players.  
The game detects real hand gestures and manages countdowns, scoring, and match flow.

---

## Features

### 🎮 Game Modes

#### Single Player (vs AI)
- Detects your hand gesture from the webcam.
- AI selects its move for each round.
- Handles scoring, round results, and match winners.

#### Two Player Mode
- Split screen: left (red) for Player 1, right (blue) for Player 2.
- Detects both gestures simultaneously.
- Competitive round flow with full scoring.

---

### 🧠 Gameplay System
- Real time gesture detection via a MediaPipe-based detector.
- Three second countdown with sound cues.
- Round result overlays with gesture symbols.
- Best of three with win by two scoring.
- Match winner state with restart and menu options.

---

### 🔊 Sound Effects
If `simpleaudio` is installed and sound files exist in `sounds/`, audio will play for:
- Countdown beeps  
- “Go” sound  
- Win/loss reactions  

Game still functions without audio files.

---

## Opponents

### AI Opponent
- AI chooses Rock, Paper, or Scissors.
- Your gesture is detected through the webcam.

### Two Human Players
- Each player uses their designated half of the frame.
- Both moves are detected and compared simultaneously.

---

## File Overview

### `main.py`
- Webcam loop  
- Mode switching  
- Countdown logic  
- UI rendering  
- Score and match handling  
- Pause and menu functionality  

### `game_logic.py`
- Valid gesture definitions  
- Round outcome logic  
- AI move selection  
- ScoreTracker class  
- Symbol formatting helpers  

### `hand_detextor.py`
- Hand detection and gesture classification  
- Prediction objects with gesture and center coordinates  

### `sounds/`
Optional audio files:
- `beep.wav`
- `cheer.wav`
- `boo.wav`
- `yay.wav`

---

## Requirements

### Python Version
- Python **3.10 recommended**

### Install Dependencies
```bash
pip install -r requirements.txt
Setup
Clone Repository
bash
Copy code
git clone https://github.com/ShrikarSwami/Rock-Paper-Scissors.git
cd Rock-Paper-Scissors
Set Python Version (pyenv)
bash
Copy code
pyenv local 3.10.12
Create & Activate Virtual Environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate
Install Packages
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
pip install simpleaudio
Running the Game
bash
Copy code
python main.py
Controls
1 — Single player

2 — Two player

p — Pause

m — Menu

q — Quit

Project Structure
arduino
Copy code
Rock-Paper-Scissors/
├── main.py
├── game_logic.py
├── hand_detextor.py
├── sounds/
│   ├── beep.wav
│   ├── cheer.wav
│   ├── boo.wav
│   └── yay.wav
├── requirements.txt
└── README.md
Future Ideas
AI difficulty settings

Animated gesture icons

Match history

Gesture calibration

Extra gesture sets (Lizard Spock)

UI animation improvements

yaml
Copy code

---

If you want this added **directly into your repo as a commit**, or want a **Wiki Home Page**, I can generate that next.
