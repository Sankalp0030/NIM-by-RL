import random
import numpy as np
class NIM:
    def __init__(self, piles):
      self.piles = piles
      self.player = 0
      self.winner = None
    
    def switch_player(self):
       self.player = 1 - self.player
    
    def action(self, pile, stones):
       if self.piles[pile] >= stones:
        self.piles[pile]-=stones
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
           self.winner = self.player
        return True
       
       return False
    
    def game_over(self):
       return all([pile == 0 for pile in self.piles])
    
    def winner(self):
       if self.game_over():
          return self.player
       else:
          return None

class QLearningAgent:
    def __init__(self, alpha=0.25, gamma=0.95, epsilon=0.35, epsilon_decay=0.99, min_epsilon=0.01):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q(self, state, action):
        return self.q.get((tuple(state), action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_value = [self.get_q(state, action) for action in actions]
        max_q = max(q_value)
        max_actions = [actions[i] for i in range(len(actions)) if q_value[i] == max_q]
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_actions = [(pile, stones) for pile in range(len(next_state)) if next_state[pile] > 0 for stones in range(1, next_state[pile] + 1)]
        
        if next_actions:
            max_q = max([self.get_q(next_state, next_action) for next_action in next_actions])
        else:
            max_q = 0  # No future reward if there are no actions

        self.q[(tuple(state), action)] = old_q + self.alpha * (reward + self.gamma * max_q - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train(n_games=0):
    agent = QLearningAgent()
    for i in range(n_games):
        # print(f"training case {i}")
        game = NIM([1, 3, 5, 7])

        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while not game.game_over():
            actions = [(pile, stones) for pile in range(len(game.piles)) for stones in range(1, game.piles[pile] +1)]
            state = game.piles.copy()
            action = agent.choose_action(state,actions)
            
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            valid_action = game.action(*action)
            if not valid_action:
                continue
            
            next_state = game.piles[:]

            if game.game_over():
                reward = -1  # Losing reward
                agent.learn(state, action, reward, next_state)
                # Reward the opponent for making the winning move
                agent.learn(last[game.player]["state"], last[game.player]["action"], 1, next_state)
            elif last[game.player]["state"] is not None:
                reward = 0  # Neutral reward
                agent.learn(last[game.player]["state"], last[game.player]["action"], reward, next_state)
        
            # print(f"Game {i}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        agent.decay_epsilon() 
    # for i,j in agent.q.items():
    #     print(f"{i} : {j}")
    return agent
       
def evaluate(agent, n_games=1000):
    wins = 0
    for _ in range(n_games):
        game = NIM([1, 3, 5, 7])
        state = game.piles[:]
        while not game.game_over():
            if game.player == 0:
                actions = [(pile, stones) for pile in range(len(game.piles)) for stones in range(1, game.piles[pile] + 1)]
                action = agent.choose_action(state, actions)
                game.action(*action)
            else:
                actions = [(pile, stones) for pile in range(len(game.piles)) for stones in range(1, game.piles[pile] + 1)]
                action = random.choice(actions)
                game.action(*action)
            
            state = game.piles[:]

        if game.winner==0:
            wins += 1
    return wins / n_games

def play(agent):
    game = NIM([1, 3, 5, 7])
    while not game.game_over():
        if game.player == 0:
            state = game.piles[:]
            actions = [(pile, stones) for pile in range(len(game.piles)) for stones in range(1, game.piles[pile] + 1)]
            action = agent.choose_action(state, actions)
            game.action(*action)
            print(f"AI took {action[1]} stones from pile {action[0]}")
        else:
            print(f"Current piles: {game.piles}")
            pile = int(input(f"Player {game.player}, choose a pile (0-{len(game.piles) - 1}): "))
            stones = int(input(f"Player {game.player}, choose number of stones to remove (1-{game.piles[pile]}): "))
            game.action(pile, stones)
    if game.winner == 0:
        print(f"AI won the Game ")
    else:
        print(f"Player {game.winner} wins!") 
     