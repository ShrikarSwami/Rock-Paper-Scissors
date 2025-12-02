# **Rock Paper Scissors CV 🖐️🤖📸**

A full webcam powered Rock Paper Scissors game.  
You throw real gestures at your camera, the system detects them, runs a countdown, compares results, updates score, and manages full matches.  
Play against an AI or challenge a friend in split-screen mode.

---

## **What This Project Does**

You stand in front of your webcam and perform Rock Paper Scissors in real life. The application:

- Detects your real hand gesture  
- Runs a 3-second countdown  
- Locks both moves at “shoot”  
- Displays symbols for the moves  
- Awards points  
- Plays sound effects (optional)  
- Runs a full “best of three, win by two” match  

In two-player mode, the screen splits left/right and detects both players’ gestures at the same time.

---

## **Features**

- Real-time hand tracking using a `HandDetector`  
- Two gameplay modes:  
  - **Single Player vs AI**  
  - **Two Player Split Screen**  
- Countdown animation with optional audio  
- Scoreboard with win-by-two logic  
- On-screen UI overlays  
- Optional sound effects using `.wav` files  
- Clean separation between CV, game logic, and UI  

---

## **Opponents**

### **AI Opponent**
- Camera reads your gesture  
- AI chooses its move via `choose_ai_move()`  
- `decide_winner()` evaluates the round  
- `ScoreTracker` handles scoring and match wins  

### **Two Human Players**
- Screen is split into left and right halves  
- Left = Player 1  
- Right = Player 2  
- Both gestures captured simultaneously  
- Countdown → results → score updates  

---

## **Folder Structure**

```
Rock-Paper-Scissors/
│
├── main.py                # Game loop, UI, countdown, state management
├── game_logic.py          # Rules, scoring, AI, win-by-two logic
├── hand_detextor.py       # Hand tracking + gesture prediction
├── sounds/
│   ├── beep.wav
│   ├── cheer.wav
│   ├── boo.wav
│   └── yay.wav
├── requirements.txt
└── README.md
```

---

## **Core Files**

### **main.py**
Handles:

- Webcam capture  
- Switching between menu, single player, and two player modes  
- Countdown timing  
- Round evaluation  
- Score updates  
- On-screen UI rendering  
- Pause, menu, and quit controls  
- Sound playback via `SoundPlayer`  

### **game_logic.py**
Includes:

- `GESTURES`  
- `ScoreTracker` (with win-by-two match logic)  
- `choose_ai_move()`  
- `decide_winner()`  
- `move_to_symbol()`  

### **hand_detextor.py**
Defines the `HandDetector` class:

- Detects hands and returns:
  - Gesture label  
  - Center location  
- Optional landmark drawing  

---

## **Installation (macOS)**

```bash
cd /path/to/Rock-Paper-Scissors
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install simpleaudio
python main.py
```

If using pyenv:

```bash
pyenv local 3.10.12
```

---

## **Installation (Windows)**

```cmd
cd C:\path\to\Rock-Paper-Scissors
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install simpleaudio
python main.py
```

---

## **Controls**

- `1` — Start single player mode  
- `2` — Start two player mode  
- `p` — Pause  
- `m` — Return to menu  
- `q` — Quit  

---

## **Troubleshooting**

### Gesture not detected
- Improve lighting  
- Move closer to the camera  
- Avoid cluttered backgrounds  

### No sound
- Ensure `.wav` files exist in the `sounds` folder  
- Verify `simpleaudio` is installed  

### Webcam not detected
- Close apps using the camera  
- Ensure camera permissions are enabled  

---

## **Future Improvements**

- Character select: Kandiddy, Dhir, JigglyDith, JigglyPathi, Skittles, Pay Gorn, Sigeon Pex  
- AI difficulty settings  
- Animated UI overlays  
- Lizard-Spock mode  
- Gesture calibration  

---

## **Final Notes**

This project mixes computer vision, gameplay design, and lots of chaotic energy.  
If you enjoyed playing or modifying it, consider starring the repo.  
Have fun battling with your webcam.
