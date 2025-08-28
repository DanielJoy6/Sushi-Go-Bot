# Sushi Go! AI Bot

This project is an Q-learning bot for playing the card game Sushi Go!
The bot slowly builds a Q-table over time than encourages optimal strategies. 
It achieves ~70% win rate against multiple opponents


## Features
- Learns strategies by self-playing millions of games
- You can set opponent strategies:
  - "Model" - uses q-table
  - "rules" + "rules2" - Good strategies modeled after my own play
  - "human" - you play!
  - "random" - random moves
- Displays lots of information on card popularity, percentage picked, winrate over training times

## Project Structure
- train.py - Trains the AI model, configurable card size, training rounds, epsilon
- play.py - allows you to play against the AI!
- game.py - Core game logic, do not touch this
- player.py - Player class logic
- q_table.pkl - saved Q-table

## Installation
1) Clone this repository
2) Install dependencies: numpy, pandas, matplotlib, tqdm
3) Train AI with train.py
4) Play against it with play.py

**NOTE** The AI is currently set to train for 100m rounds, or ~10 hours. 
It may learn optimal strategies sooner.

## Performance
- Trained for 100m+ games
- Achieves ~70% winrate vs 1, 2, or 3 opponents

## "What's the optimal strategy for destroying my family?"
- Grab dumplings early
- Don't go for wasabi if you can't get 1 round 1
- Don't bother with maki especially in higher play games; focus on reliablity

## Contributing
Please feel free to submit pull requests if you think you can improve something!

## Can I use this?
Sure! Just please credit me if you want to show it off
