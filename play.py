"""
Logic to play 1 round against the AI, which makes what it thinks are optimal moves
"""
import random

from game import Game
from player import Player

NUM_CARDS = 8

AI1 = Player("AI1", "model", None, epsilon=0.01, alpha=0.05, gamma=0.9)  # Should be best
AI2 = Player("Human", "player", None)
players = [AI1, AI2]
evaluation_wins = {}
for player in players:
    evaluation_wins[player.name] = []
evaluation_wins["Ties"] = []
game = Game(num_cards=NUM_CARDS, players=players, print_info=True)
random.shuffle(game.players)
game.deal_cards()
game.play_round()
game.score_round()
finalscores = game.ending()
if finalscores[1] > finalscores[0]:
    print("The AI beat you", finalscores[1], "points to", finalscores[0])
else:
    print("You beat the AI", finalscores[1], "points to", finalscores[0])
