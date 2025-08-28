"""
This is the main file for training model / building out q-table
"""

import pickle

# trunk-ignore-all(pylint/E0401)
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from game import Game
from player import Player

NUM_CARDS = 5
NUM_ROUNDS = 1_000_000
NUM_ROUNDS_PER_GAME = 1
INCREMENT = 100000
NUM_SIMULATION_GAMES = 10000
EPSILON = 0.90

print("Number of Cards:", NUM_CARDS, "Rounds:", NUM_ROUNDS)
AI1 = Player(
    "AI1", "model", None, epsilon=EPSILON, alpha=0.05, gamma=0.9
)  # Should be best
AI2 = Player("Random1", "random", None)
AI3 = Player("Random2", "random", None)
# AI4 = Player("Random3", "random", None)
players = [AI1, AI2, AI3]  # , AI4]
evaluation_wins = {}
for player in players:
    evaluation_wins[player.name] = []
evaluation_wins["Ties"] = []

# @title Simulate Games

print("Epsilon =", EPSILON)

epsilon_decay = [max(0.1, EPSILON * (0.9999995**i)) for i in range(NUM_ROUNDS)]


def plot_data(temp_evaluation_wins, temp_players, temp_increment):
    """Plots Wins from evaluations games over time, for all players"""
    games = list(
        range(
            0,
            temp_increment * len(temp_evaluation_wins[temp_players[0].name]),
            temp_increment,
        )
    )
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "gray"]
    for temp_player in temp_players:
        plt.plot(
            games,
            temp_evaluation_wins[temp_player.name],
            label=temp_player.name,
            color=colors[_],
        )
    plt.plot(games, temp_evaluation_wins["Ties"], label="Ties", color="black")
    plt.legend()
    plt.title("Wins")
    plt.ylabel("Wins")
    plt.xlabel("Training rounds")
    plt.show()


try:
    start = time.time()
    SAVE_COUNTER = 0
    game = Game(num_cards=NUM_CARDS, players=players, print_info=False)
    for numGames in tqdm(range(NUM_ROUNDS)):
        game.reset()  # or game = Game(...) if you still instantiate
        AI1.epsilon = epsilon_decay[numGames]
        for _ in range(NUM_ROUNDS_PER_GAME):
            random.shuffle(game.players)
            game.deal_cards()
            game.play_round()
            game.score_round()
        if (numGames + 1) % INCREMENT == 0:
            average_update = np.mean(AI1.q_updates[-INCREMENT:])
            AI1.q_updates = []
            print("\nAverage Q-table update:", average_update)
            print("Epsilon:", AI1.epsilon)
            print(
                "numGames:", (numGames + 1), "Q-table size:", len(AI1.q_table), end=" "
            )

            AI1.epsilon = 0
            evaluation_set_wins = {player.name: 0 for player in players}
            evaluation_set_wins["Ties"] = 0
            for _ in range(NUM_SIMULATION_GAMES):
                game2 = Game(
                    num_cards=NUM_CARDS, players=players, print_info=False, update=False
                )
                for _ in range(NUM_ROUNDS_PER_GAME):
                    random.shuffle(game2.players)
                    game2.deal_cards()
                    game2.play_round()
                    game2.score_round()
                finalpoints = game2.ending()
                won_player = finalpoints.index(max(finalpoints))
                finalpoints.sort(reverse=True)
                if finalpoints[0] != finalpoints[1]:
                    evaluation_set_wins[players[won_player].name] += 1
                else:
                    evaluation_set_wins["Ties"] += 1
            print("WINS:", evaluation_set_wins)
            for player in players:
                evaluation_wins[player.name].append(evaluation_set_wins[player.name])
            evaluation_wins["Ties"].append(evaluation_set_wins["Ties"])

            FILENAME = "q_table.pkl" if SAVE_COUNTER == 0 else "q_table1.pkl"
            with open(FILENAME, "wb") as file:
                pickle.dump(AI1.q_table, file)
            SAVE_COUNTER = 1 - SAVE_COUNTER
            print("\nDone Saving\n")

            AI1.epsilon = EPSILON

except KeyboardInterrupt:
    plot_data(evaluation_wins, players, INCREMENT)

end = time.time()
print("Time taken:", (end - start))
# @title Print Stats
card_types = [
    "Dumpling",
    "EggNigiri",
    "Maki1",
    "Maki2",
    "Maki3",
    "SalmonNigiri",
    "Sashimi",
    "SquidNigiri",
    "Tempura",
    "Wasabi",
]

Popularities = [{card: 0 for card in card_types} for _ in range(NUM_CARDS)]
percents = [[0 for _ in range(10)] for _ in range(NUM_CARDS)]
Seen = [{card: 0 for card in card_types} for _ in range(NUM_CARDS)]

q_table = AI1.q_table
print("Q-table size:", len(q_table))
# column_width = 4  # For alignment

for state, q_values in q_table.items():
    hand_vector = state[:10]  # First 10 elements of tuple = hand counts
    total_cards = sum(hand_vector)
    if total_cards == 0 or total_cards > NUM_CARDS:
        continue  # Skip invalid or oversized hands

    # Reconstruct the list of available cards in the hand
    hand = []
    for i, count in enumerate(hand_vector):
        hand.extend([card_types[i]] * count)
        Seen[total_cards - 1][card_types[i]] += count

    if not hand:
        continue  # Avoid edge case where hand is empty

    # Find the index of the highest Q-value among all possible actions
    max_index = np.argmax(q_values)
    if max_index >= len(hand):
        continue  # Avoid index error

    chosen_card = hand[max_index]
    Popularities[total_cards - 1][chosen_card] += 1

# Print percentages and seen counts together
for hand_size, popularity in enumerate(Popularities):
    print(f"\nHand Size {hand_size + 1}:")
    print("Card\t\tTimes Seen\tTimes Played\t% Played")

    total_played = sum(popularity.values())
    for i, card in enumerate(card_types):
        times_played = popularity[card]
        times_seen = Seen[hand_size][card]
        percent_played = (
            round(times_played / total_played, 4) if total_played > 0 else 0.0
        )

        print(
            f"{card.ljust(14)}\t"
            f"{str(times_seen).ljust(10)}\t"
            f"{str(times_played).ljust(10)}\t"
            f"{str(percent_played).ljust(10)}"
        )

        # Store percentage correctly
        percents[hand_size][i] = 100 * percent_played

    print("Total Played:", total_played)

# Plot
plot_colors = [
    "Blue",
    "Green",
    "Red",
    "Black",
    "Brown",
    "Yellow",
    "Pink",
    "Purple",
    "Orange",
    "lightblue",
]
for i in range(10):
    values = [percents[size][i] for size in range(NUM_CARDS)]
    plt.plot(range(1, NUM_CARDS + 1), values, label=card_types[i], color=plot_colors[i])
plt.xlabel("Hand Size")
plt.ylabel("Percentage picked")
plt.title(f"{NUM_CARDS} Cards")
plt.legend()
plt.show()

# Popularity vs seen
for i, card in enumerate(card_types):
    popularity_values = [percents[size][i] for size in range(NUM_CARDS)]
    seen_totals = [sum(Seen[size].values()) for size in range(NUM_CARDS)]
    seen_values = [
        (100 * Seen[size][card] / seen_totals[size]) if seen_totals[size] > 0 else 0
        for size in range(NUM_CARDS)
    ]
    # plt.plot(range(1, NUM_CARDS + 1), popularity_values,
    # label=f"{card} Popularity", linestyle='-', color=colors[i])
    plt.plot(
        range(1, NUM_CARDS + 1),
        seen_values,
        label=f"{card} Seen",
        linestyle="--",
        color=plot_colors[i],
    )

plt.xlabel("Hand Size")
plt.ylabel("Percentage (%)")
plt.title(f"{NUM_CARDS} Cards - How much cards are seen")
plt.legend(ncol=2)
plt.show()

# @title How often cards are picked vs just seen
table_rows = []
for size in range(NUM_CARDS):
    total_pop = sum(Popularities[size].values())
    total_seen = sum(Seen[size].values())
    for card in card_types:
        pop_count = Popularities[size][card]
        seen_count = Seen[size][card]
        pop_percent = (100 * pop_count / total_pop) if total_pop > 0 else 0
        seen_percent = (100 * seen_count / total_seen) if total_seen > 0 else 0
        ratio = (pop_percent / seen_percent) if seen_percent > 0 else 0
        table_rows.append(
            [
                size + 1,
                card,
                round(pop_percent, 2),
                round(seen_percent, 2),
                round(ratio, 2),
            ]
        )

df = pd.DataFrame(
    table_rows,
    columns=["Hand Size", "Card Type", "Popularity %", "Seen %", "Pop/Seen Ratio"],
)
df.sort_values("Pop/Seen Ratio")
print(df.to_string(index=False))
