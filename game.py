"""
Game file for all the game logic
"""
# trunk-ignore-all(pylint/E0401)

import numpy as np

dumplingPoints = [0, 1, 3, 6, 10, 15, 15, 15, 15, 15, 15, 15]

class Game:
    """Game logic for playing 1 round of the SushiGo game"""
    def __init__(self, num_cards, players, print_info, update=True):
        """Initialize variables"""
        self.num_cards = num_cards
        self.num_players = len(players)
        self.deck = self.create_deck()
        self.round = 1
        self.players = players
        self.original_players = players
        self.scores = [0 for x in range(self.num_players)]
        self.previous_scores = [0 for _ in range(self.num_players)]
        self.print_info = print_info
        self.update = update
        self.card_type_indices = {
            "Wasabi": 9,
            "Tempura": 8,
            "SquidNigiri": 7,
            "Sashimi": 6,
            "SalmonNigiri": 5,
            "Maki3": 4,
            "Maki2": 3,
            "Maki1": 2,
            "EggNigiri": 1,
            "Dumpling": 0,
        }

    def reset(self):
        """Resets variables before next game"""
        self.deck = self.create_deck()
        self.round = 1
        self.players = self.original_players  # restore fresh state
        self.scores = [0 for _ in range(self.num_players)]
        self.previous_scores = [0 for _ in range(self.num_players)]

    def encode_game_state(self, game_state):
        """Turns gamestate into a list of integers to be passed into qtable"""
        hand_encoded = self.encode_cards_as_number(game_state["hand"])
        played_encoded = self.encode_cards_as_number(game_state["played_cards"])
        return tuple(np.concatenate([hand_encoded, played_encoded]).tolist())

    def encode_cards_as_number(self, cards):
        """Smaller function to turn cards into list of integers"""
        encoding = [0] * 10
        for card in cards:
            if card in self.card_type_indices:
                encoding[self.card_type_indices[card]] += 1
        return encoding

    def create_deck(self):
        """Generates a deck of cards, randomly shuffles"""
        card_types = {
            "Tempura": 14,
            "Sashimi": 14,
            "Dumpling": 14,
            "Maki1": 6,
            "Maki2": 12,
            "Maki3": 8,
            "SalmonNigiri": 10,
            "SquidNigiri": 5,
            "EggNigiri": 5,
            "Wasabi": 6,
        }
        cards = [name for name, count in card_types.items() for _ in range(count)]
        np.random.shuffle(cards)
        return cards

    def deal_cards(self):
        """Deals out cards from deck to players' hands"""
        for player in self.players:
            player.hand = [self.deck.pop() for _ in range(self.num_cards)]
            player.hand.sort()

    def play_round(self):
        """Plays a single round of the game"""
        for i in range(self.num_cards):
            for player in self.players:
                game_state = {
                    "hand": list(player.hand),
                    "played_cards": list(player.played_cards),
                }
                encoded_state = self.encode_game_state(game_state)
                if self.update:
                    player.choose_card_ai(encoded_state, True)
                else:
                    player.choose_card_ai(encoded_state)
            if i != self.num_cards - 1:
                hands = [player.hand for player in self.players]
                for j, player in enumerate(self.players):
                    player.hand = hands[(j + 1) % len(self.players)]

    def score_round(self):
        """Scores each player based on players' hand"""
        makis = [0 for x in range(len(self.players))]
        counter = 0
        for player in self.players:
            score = 0
            mydict = {
                "Tempura": 0,
                "Sashimi": 0,
                "Dumpling": 0,
                "Maki2": 0,
                "Maki3": 0,
                "Maki1": 0,
                "SalmonNigiri": 0,
                "SquidNigiri": 0,
                "EggNigiri": 0,
                "Wasabi": 0,
            }
            for card in player.played_cards:
                if (card in ["Tempura", "Sashimi", "Dumpling",
                             "Maki1", "Maki2", "Maki3", "Wasabi"]):
                    mydict[card] += 1
                elif (card in ["SalmonNigiri", "SquidNigiri", "EggNigiri"]):
                    if mydict["Wasabi"] > 0:
                        mydict[card] += 3
                        mydict["Wasabi"] -= 1
                        # score += 2
                    else:
                        mydict[card] += 1
            # score -= mydict["Wasabi"] #Penalize leftover Wasabi
            score += (int(mydict["Tempura"] / 2) * 5) - (mydict["Tempura"] % 2)
            score += (int(mydict["Sashimi"] / 3) * 10) - (mydict["Sashimi"] % 3)

            score += mydict["EggNigiri"]
            score += 2 * mydict["SalmonNigiri"]
            score += 3 * mydict["SquidNigiri"]
            score += dumplingPoints[mydict["Dumpling"]]

            makiscore = 1 * mydict["Maki1"] + 2 * mydict["Maki2"] + 3 * mydict["Maki3"]
            makis[counter] = makiscore
            self.scores[counter] += score
            player.played_cards = []
            counter += 1
            if self.print_info:
                print("Player", player.name, "played:", mydict)
        maki_points = [0 for x in range(len(self.players))]
        sorted_scores = sorted(enumerate(makis), key=lambda x: x[1], reverse=True)
        first_place_score = sorted_scores[0][1]
        first_place_indices = [
            idx for idx, score in sorted_scores if score == first_place_score
        ]
        points_first = 6 // len(first_place_indices)
        for idx in first_place_indices:
            maki_points[idx] += points_first
        second_place_scores = [
            score for idx, score in sorted_scores if score != first_place_score
        ]
        if second_place_scores:
            second_place_score = second_place_scores[0]
            second_place_indices = [
                idx for idx, score in sorted_scores if score == second_place_score
            ]
            points_second = 3 // len(second_place_indices)
            for idx in second_place_indices:
                maki_points[idx] += points_second
        for i, points in enumerate(maki_points):
            self.scores[i] += points
        if self.update:
            for i, player in enumerate(self.players):
                if player.strategy == "model":
                    round_reward = self.scores[i] - self.previous_scores[i]
                    player.update_q_table(round_reward)
        self.round += 1
        self.previous_scores = self.scores[:]

    def ending(self):
        """Returns the scores as an array for game ending"""
        return self.scores
