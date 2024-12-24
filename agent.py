# new
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, Dueling_Noisy_QNet
from helper import plot
import math
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
# LR = 0.001 -> n_games = 25 avg score = 0.03
# LR = 0.01 -> n_games = 25 avg score = 0.03

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class RainbowAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Start with high exploration
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        self.gamma = 0.99  # Discount rate
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.model = Dueling_Noisy_QNet(11, 256, 3)
        self.target_model = Dueling_Noisy_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Explore
        else:
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()

    def train_long_memory(self):
        if len(self.memory.buffer) > BATCH_SIZE:
            mini_batch, indices, weights = self.memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*mini_batch)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float)
            next_states = torch.tensor(next_states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.bool)
            weights = torch.tensor(weights, dtype=torch.float)

            # Compute Q targets using target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                next_actions = torch.argmax(next_q_values, dim=1)
                next_q_values = self.target_model(next_states)[torch.arange(BATCH_SIZE), next_actions]
                targets = rewards + (self.gamma ** self.n_step) * next_q_values * ~dones

            # Compute predicted Q values
            predicted_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Calculate loss
            td_errors = targets - predicted_q_values
            loss = (weights * td_errors.pow(2)).mean()

            # Optimize the model
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()

            # Update priorities in the replay buffer
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()