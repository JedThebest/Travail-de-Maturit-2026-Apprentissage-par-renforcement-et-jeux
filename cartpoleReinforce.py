import torch 
import torch.nn as nn 
import torch.optim as optim 
import random 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import gymnasium as gym 
 
env = gym.make('CartPole-v1') 
 
class PolicyNetwork(nn.Module): 
 
    def __init__(self,state_dim:int, action_dim:int): 
        super().__init__() 
 
        self.network = nn.Sequential( 
            nn.Linear(state_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim), 
            nn.Softmax(dim=-1) 
        ) 
 
    def forward(self, x): 
        return self.network(x) 
 
class Agent: 
 
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99): 
     
        self.gamma = gamma 
 
        self.policy = PolicyNetwork(state_size, action_size) 
 
        self.optimizer = optim.Adam( 
            self.policy.parameters(), 
            lr=lr 
        ) 
        self.state_dim = state_size 
        self.action_dim = action_size 
 
    def get_action(self, state): 
 
        state_tensor = torch.tensor(state, dtype=torch.float32) 
 
        probs = self.policy(state_tensor) 
 
        distribution = torch.distributions.Categorical(probs) 
 
        action = distribution.sample() 
 
        log_prob = distribution.log_prob(action) 
 
        return action.item(), log_prob 
 
    def train(self,env:gym.Env, episodes=1000): 
 
        scores = [] 
 
        for episode in tqdm(range(episodes)): 
         
            state, _ = env.reset() 
 
            log_probs = [] 
            rewards = [] 
 
            done = False 
 
            while not done: 
 
                action, log_prob = self.get_action(state) 
 
                next_state, reward, terminated, truncated, _ = env.step(action) 
 
                log_probs.append(log_prob) 
                rewards.append(reward) 
 
                state = next_state

                done = terminated or truncated
 
            returns = [] 
 
            G = 0 
 
            for reward in reversed(rewards): 
                G = reward + self.gamma * G 
                returns.insert(0, G) 
 
            returns = torch.tensor(returns, dtype=torch.float32) 
             
            # Normalisation des retours => éviter les gradients trop grands
 
            if len(returns) > 1: 
                returns = (returns - returns.mean()) / (returns.std() + 1e-8) 
 
            loss = 0 
 
            for log_prob, G in zip(log_probs, returns): 
                loss += -log_prob * G 
 
            self.optimizer.zero_grad() 
 
            loss.backward() 
 
            self.optimizer.step() 
 
            scores.append(sum(rewards)) 
 
        return scores 
 
state_size = env.observation_space.shape[0] 
action_size = env.action_space.n 

agent = Agent(state_size, action_size) 

scores = agent.train(env, 2000) 
 
plt.plot(scores)
plt.xlabel("Épisode")
plt.ylabel("Récompense")
plt.title("REINFORCE - Cartpole")
plt.show()

def test(agent, episodes=5):

    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(episodes):

        state, _ = env.reset()

        done = False
        score = 0

        while not done:

            state_tensor = torch.tensor(state, dtype=torch.float32)

            probs = agent.policy(state_tensor)

            action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            score += reward

        print(f"Episode {episode + 1} : {score}")

    env.close()

test(agent)