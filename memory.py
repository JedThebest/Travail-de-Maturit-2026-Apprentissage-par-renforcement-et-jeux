import random, torch
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity:int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # On empile et convertit directement ici en une seule opération native
        return (
            torch.stack(states).float(),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards),
            torch.stack(next_states).float(),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)