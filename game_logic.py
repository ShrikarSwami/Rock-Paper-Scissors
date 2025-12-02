"""
Core game rules for Rock, Paper, Scissors.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

GESTURES = ("rock", "paper", "scissors")
WIN_MAP = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def choose_ai_move() -> str:
    """Return a random move for the AI opponent."""
    return random.choice(GESTURES)


def decide_winner(p1_move: str, p2_move: str) -> str:
    """
    Decide the round outcome.

    Returns:
        "p1" if player 1 wins,
        "p2" if player 2 wins,
        "tie" if moves are equal,
        "invalid" if either move is unknown.
    """
    if p1_move not in GESTURES or p2_move not in GESTURES:
        return "invalid"
    if p1_move == p2_move:
        return "tie"
    return "p1" if WIN_MAP[p1_move] == p2_move else "p2"


def move_to_symbol(move: str) -> str:
    """Map a gesture string to an emoji-like symbol for on-screen display."""
    return {"rock": "✊", "paper": "🤚", "scissors": "✌️"}.get(move, "?")


@dataclass
class ScoreTracker:
    target: int = 3  # first to 3 wins
    lead: int = 0
    p1: int = 0
    p2: int = 0

    def apply_round(self, result: str) -> None:
        """Update the score for a finished round."""
        if result == "p1":
            self.p1 += 1
        elif result == "p2":
            self.p2 += 1

    def winner(self) -> Optional[str]:
        """
        Check if the match is over.

        A player wins when they reach `target` points and lead by `lead`.
        """
        if max(self.p1, self.p2) >= self.target and abs(self.p1 - self.p2) >= self.lead:
            return "p1" if self.p1 > self.p2 else "p2"
        return None

    def reset(self) -> None:
        self.p1 = 0
        self.p2 = 0

    def as_text(self, labels: Tuple[str, str]) -> str:
        """Return a short scoreboard line."""
        return f"{labels[0]} {self.p1} - {self.p2} {labels[1]}"
