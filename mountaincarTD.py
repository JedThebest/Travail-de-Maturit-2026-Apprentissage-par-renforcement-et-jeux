
import numpy as np
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")


# paramètres pour l'approximation
pos_min = -1.2
pos_max = 0.6
vel_min = -0.07
vel_max = 0.07
total_states = 50

pos_bins = np.linspace(pos_min, pos_max, total_states)
vel_bins = np.linspace(vel_min, vel_max, total_states)

def approximise(position, velocity):
    
    # np.digitize renvoie un indice entre 1 et total_states, on soustrait 1 pour avoir 0-19
    pos_approximate = int(np.clip(np.digitize(position, pos_bins) - 1, 0, total_states - 1))
    vel_approximate = int(np.clip(np.digitize(velocity, vel_bins) - 1, 0, total_states - 1))
    return pos_approximate, vel_approximate

Q = np.zeros((total_states, total_states, env.action_space.n)) # pos x vel x actions

# hyperparamètres
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.05

# historique
hist_done = []

for episode in tqdm.tqdm(range(int(1e4))):
    current_state, _ = env.reset()
    current_state = approximise(current_state[0], current_state[1])
    done, truncated = False, False

    while not (done or truncated):
        
        if np.random.random() < epsilon: # exploration
            action = env.action_space.sample()
        else: # exploitation
            action = np.argmax(Q[current_state[0]][current_state[1]])

        next_state, reward, done, truncated , _ = env.step(action)
        next_state = approximise(next_state[0], next_state[1])

        Q[current_state[0]][current_state[1]][action] += alpha*(reward + gamma*np.max(
            Q[next_state[0]][next_state[1]]
        )- Q[current_state[0]][current_state[1]][action])

        current_state = next_state
        
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    if hist_done == []:
        hist_done.append(int(done))
    else:
        hist_done.append(int(done)+hist_done[-1])
print(Q)

# Visualisation du modèle
new_env = gym.make("MountainCar-v0", render_mode='human')
done, truncated = False, False

state, _ = new_env.reset()
state = approximise(state[0], state[1])

while not (done or truncated):
    action = np.argmax(Q[state[0]][state[1]])
    next_state, _, done, truncated, _ = new_env.step(action)
    state = approximise(next_state[0], next_state[1])


plt.figure(figsize=(15,6))

plt.subplot(1,2,1) # évolution du nombre de fois que l'agent a réussi
plt.plot(hist_done, c='green', lw=3)
plt.xlabel("épisodes")
plt.ylabel("Nombre de fois que l'agent a réussi à atteindre le drapeau")
plt.title("évolution du nombre de fois que l'agent a réussi à atteindre le drapeau")
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1,2,2) # heatmap des valeurs d'états
values = np.max(Q, axis=2)

values_t = values.T

# Mettre les valeurs de 0 en gris
masked_values = np.ma.masked_equal(values_t, 0)
current_cmap = plt.cm.plasma.copy()
current_cmap.set_bad(color='lightgray')

plt.imshow(masked_values, cmap=current_cmap, origin='lower',
           extent=[pos_min, pos_max, vel_min, vel_max], aspect='auto')
plt.colorbar(label="Valeur de l'état")
plt.xlabel('Position')
plt.ylabel('Vitesse')
plt.title("Visualisation de la fonction de valeur")

plt.tight_layout()
plt.show()
