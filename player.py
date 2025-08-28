"""Player class"""

import pickle
import random


class Player:
    """Player class for each player in the game"""

    def __init__(self, name, strategy, model, epsilon=0.9, alpha=0.3, gamma=0.8):
        self.name = name
        self.hand = []
        self.played_cards = []
        self.strategy = strategy
        self.ai_model = model
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_updates = []
        if strategy == "model":
            try:
                with open("q_table.pkl", "rb") as file:
                    self.q_table = pickle.load(file)
                print("Successfully loaded in Q Table")
            except EOFError:
                self.q_table = {}
                print("Unsuccessfully loaded in Q Table - EOFError")
            except FileNotFoundError:
                self.q_table = {}
                print("Unsuccessfully loaded in Q Table - FileNotFoundError")
        else:
            self.q_table = {}
        self.state_action_pairs = []
        self.cumulative_reward = 0

    def choose_card_ai(self, game_state, update=False):
        """Algorithm for choosing a card, based on strategy"""
        chosen_card = ""
        if self.strategy == "model":
            # Convert encoded NumPy array into hashable tuple
            state = tuple(game_state)

            # Collect first occurrence index of each unique card
            indices = {}
            for i, card in enumerate(self.hand):
                if card not in indices:
                    indices[card] = i
            index_values = list(indices.values())

            # Get Q-values (initialize if state not present)
            q_vals = self.q_table.setdefault(state, [0.0] * 10)

            # Choose action: exploration or exploitation
            if random.random() < self.epsilon:
                chosen_card_index = random.choice(index_values)
            else:
                # Select best action from available indices only
                chosen_card_index = max(index_values, key=lambda i: q_vals[i])
            # Remove and record chosen card
            chosen_card = self.hand.pop(chosen_card_index)

            # Store state-action pair for update
            if update:
                self.state_action_pairs.append((state, chosen_card_index))

        elif self.strategy == "random":
            chosen_card_index = random.randrange(len(self.hand))
            chosen_card = self.hand.pop(chosen_card_index)

        elif self.strategy == "rules":
            priority_list = [
                "SquidNigiri",
                "Sashimi",
                "Wasabi",
                "Tempura",
                "SalmonNigiri",
                "Maki3",
                "Dumpling",
                "Maki2",
                "EggNigiri",
                "Maki1",
            ]
            for card in priority_list:
                if card in self.hand:
                    self.hand.remove(card)
                    chosen_card = card
        elif self.strategy == "rules2":
            priority_list = [
                "Dumpling",
                "SquidNigiri",
                "Tempura",
                "SalmonNigiri",
                "Sashimi",
                "Maki3",
                "Maki2",
                "EggNigiri",
                "Wasabi",
                "Maki1",
            ]
            chosen_card_index = next(
                (
                    index
                    for index, card in enumerate(self.hand)
                    if card in priority_list
                ),
                0,
            )
            chosen_card = self.hand.pop(chosen_card_index)

        elif self.strategy == "worst":
            priority_list = [
                "Wasabi",
                "Maki1",
                "EggNigiri",
                "Maki2",
                "Maki3",
                "Sashimi",
                "SalmonNigiri",
                "Dumpling",
                "Tempura",
                "SquidNigiri",
            ]
            chosen_card_index = next(
                (
                    index
                    for index, card in enumerate(self.hand)
                    if card in priority_list
                ),
                0,
            )
            chosen_card = self.hand.pop(chosen_card_index)

        elif self.strategy in ("human", "player"):
            print("You've played:")
            print("Hand:")
            for _ in range(len(self.hand)):
                print(str(_) + ") " + str(self.hand[_]))
            print("Which card would you like to play?: ")
            chosen_card_index = int(input())
            chosen_card = self.hand.pop(chosen_card_index)

        self.played_cards.append(chosen_card)

    def update_q_table(self, reward):
        """Add states to q_table, update rewards"""
        for state, action in self.state_action_pairs:
            q_values = self.q_table.setdefault(state, [0.0] * 10)
            current_q = q_values[action]
            max_future_q = max(q_values)
            new_q = current_q + self.alpha * (
                reward + self.gamma * max_future_q - current_q
            )
            q_values[action] = new_q
            self.q_updates.append(abs(new_q - current_q))
        self.state_action_pairs.clear()
