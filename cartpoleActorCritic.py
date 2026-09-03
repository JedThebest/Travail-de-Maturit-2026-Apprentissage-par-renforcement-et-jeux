import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import gymnasium as gym  
  
env = gym.make('CartPole-v1')  
 
class Actor(nn.Module): 

    def __init__(self, state_dim, action_dim): 
        super().__init__() 
 
        self.network = nn.Sequential( 
            nn.Linear(state_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim), 
            nn.Softmax(dim=-1) 
        ) 
 
    def forward(self, x): 
        return self.network(x) 
 
class Critic(nn.Module): 

    def __init__(self, state_dim): 
        super().__init__() 
 
        self.network = nn.Sequential( 
            nn.Linear(state_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        ) 
 
    def forward(self, x): 
        return self.network(x) 
 
class Agent: 
 
    def __init__(self,state_size,action_size,lr_actor=0.001,lr_critic=0.01,gamma=0.99): 

        self.gamma = gamma 
         
        self.actor = Actor( 
            state_size, 
            action_size 
        ) 
 
        self.critic = Critic( 
            state_size
        ) 
 
        self.actor_optimizer = optim.Adam( 
            self.actor.parameters(), 
            lr=lr_actor 
        ) 
 
        self.critic_optimizer = optim.Adam( 
            self.critic.parameters(), 
            lr=lr_critic 
        ) 
 
        self.state_size = state_size 
        self.action_size = action_size 
 
    def get_action(self, state): 
 
        state_tensor = torch.tensor(state, dtype=torch.float32)  
          
        probs = self.actor(state_tensor)  
     
        distribution = torch.distributions.Categorical(probs)  
     
        action = distribution.sample()  
     
        log_prob = distribution.log_prob(action) 
 
        value = self.critic(state_tensor) 
     
        return action.item(), log_prob, value 
 
    def train(self, env:gym.Env, episodes=1000): 
 
        scores = [] 
         
        for episode in tqdm(range(episodes)): 

            state, _ = env.reset()

            score = 0 
            done = False 
 
            while not done: 
 
                action, log_prob, value = self.get_action(state) 
 
                next_state, reward, terminated, truncated, _ = env.step(action) 
 
                next_state_tensor = torch.tensor(
                    next_state,
                    dtype=torch.float32
                )

                next_value = self.critic(next_state_tensor) 
 
                score += reward 
 
                done = terminated or truncated 
 
                if done: 

                    target = torch.tensor(
                        [[reward]],
                        dtype=torch.float32
                    )

                else: 

                    target = (
                        torch.tensor(
                            [[reward]],
                            dtype=torch.float32
                        )
                        + self.gamma * next_value
                    ) 
 
                # Avantage 
 
                advantage = (
                    target - value
                ).detach() 
 
                # Loss Actor 
 
                actor_loss = (
                    -log_prob * advantage
                ) 
 
                # Loss Critic 
 
                critic_loss = (
                    target - value
                ).pow(2).mean() 
 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step() 
 
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step() 
 
                state = next_state 
 
            scores.append(score) 
 
        return scores 
 
state_size = env.observation_space.shape[0]  
action_size = env.action_space.n  
 
agent = Agent(state_size, action_size)  
 
scores = agent.train(env, 2000)  
  
plt.plot(scores) 
plt.xlabel("Épisode") 
plt.ylabel("Récompense") 
plt.title("Actor Critic - Cartpole") 
plt.show()

def test(agent, episodes=5):

    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(episodes):

        state, _ = env.reset()

        done = False
        score = 0

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            probs = agent.actor(state_tensor)

            action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            score += reward

        print(f"Episode {episode + 1} : {score}")

    env.close()

test(agent)