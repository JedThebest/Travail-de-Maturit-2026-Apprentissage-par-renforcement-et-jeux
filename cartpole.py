import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import tqdm

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        x = self.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = "cpu"

        self.gamma = 0.95          
        self.epsilon = 1.0         
        self.epsilon_min = 0.01    
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001 
        self.batch_size = 64       

        self.memory = deque(maxlen=2000)  

        self.policy_net = DQN(input_size=self.state_size, output_size=self.action_size).to(self.device)
        self.target_net = DQN(input_size=self.state_size, output_size=self.action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values, dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(experience[4]) for experience in minibatch]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            target_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (self.gamma * target_q * (1 - dones))

        current_q = self.policy_net(states).gather(1, actions)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_cartpole():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 2000
    target_update_frequency = 10
    scores = []
    solved_score = 495  

    for episode in tqdm.tqdm(range(episodes)):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output
            

        score = 0
        done = False

        while not done:
            action = agent.select_action(state)

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, done, truncated, _ = step_output
                if truncated:
                    done = True
            elif len(step_output) == 4:
                next_state, reward, done, _ = step_output
            else:
                raise ValueError(f"Unexpected number of return values from env.step(): {len(step_output)}")


            agent.remember(state, action, reward, next_state, done)

            agent.train_step()

            state = next_state
            score += reward

        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        scores.append(score)
        mean_score = np.mean(scores[-100:])

        
        if mean_score >= solved_score and episode >= 400:
            #print(f"\nEnvironment solved in {episode + 1} episodes! Average Score: {mean_score:.2f}")
            break

    env.close()
    return agent, scores

print("Training the agent...")
agent, scores = train_cartpole()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(scores, label='Score per Episode')
moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
plt.plot(moving_avg, label='Average Score (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title("DQN sur CartPole-v1 progression de l'entrainement")
plt.legend()
plt.show()

print("\nStarting visualization...")
def visualize_agent(agent:DQNAgent, episodes=5):
    """
    Visualise l'agent DQN entraîné sur CartPole-v1.

    Args:
        agent: agent DQN entraîné
        episodes: nombre d'épisodes à visualiser
    """
    env = gym.make("CartPole-v1", render_mode="human")

    # Sauvegarde l'epsilon actuel
    old_epsilon = agent.epsilon

    # Pas d'exploration pendant la visualisation
    agent.epsilon = 0.0

    scores = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        score = 0

        while not done:
            # Action choisie uniquement selon le réseau
            action = agent.select_action(state, evaluate=True)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            state = next_state
            score += reward

        scores.append(score)

        print(f"Episode {episode + 1}/{episodes} - Score : {score:.0f}")

    env.close()

    # Restaurer epsilon
    agent.epsilon = old_epsilon

    print(f"\nScore moyen : {np.mean(scores):.2f}")
    print(f"Meilleur score : {np.max(scores):.0f}")

    return scores

visualize_agent(agent)