import numpy as np
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)

Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_decay = 0.9995
epsilon_min = 0.05

episodes = int(1e4)

q_state_9 = [Q[9].copy()]

scores = [0]

for episode in tqdm.tqdm(range(episodes)):
    current_state,_ = env.reset()
    while True:
        if epsilon > np.random.random(): # explorer
            action = env.action_space.sample()
        else: # exploiter
            action = np.argmax(Q[current_state])
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Mise à jour de la Q-table sous Q-Learning
        Q[current_state][action] += alpha*(reward + gamma*np.max(Q[next_state])- Q[current_state][action])

        if terminated or truncated:
            scores.append(scores[-1]+reward)
            break

        current_state = next_state

    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    q_state_9.append(Q[9].copy())

print(Q)

colors = plt.cm.jet(np.linspace(0, 1, env.observation_space.n + 1))
plt.figure(figsize=(15,8))
values = np.max(Q,1)

plt.subplot(2,2,1) # L'IA m'a aidé pour ce graphique

# Redimensionnement du vecteur en matrice 4 lignes x 4 colonnes
grid_values = values.reshape(4, 4)

plt.imshow(grid_values, cmap='plasma', aspect='equal')
plt.title("Heatmap finale des valeurs (Grille 4x4)")

# Ajout des valeurs numériques dans chaque case
for i in range(4):
    for j in range(4):
        val = grid_values[i, j]
        text_color = "white" if val < 0.4 else "black"
        # plt.text(x, y, texte, ...)
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
        # placer l'index de l'état
        n = j + 4*i
        plt.text(j+0.4, i+0.4, f"{n}", ha="center", va="center", color=text_color, fontsize=8)

# Configuration des axes et de la barre de couleur
plt.xticks(range(4))
plt.yticks(range(4))
plt.xlabel("Colonnes")
plt.ylabel("Lignes")
plt.colorbar(label="Valeur de l'état")

plt.subplot(2,2,2) # analyse de l'état 9 (aidé par l'IA)
q_state_9 = np.array(q_state_9)
for a in range(env.action_space.n):
    plt.plot(q_state_9[:,a], c= colors[a*3], label=f"valeur d'action {a}")
plt.title("Évolution des valeurs d'actions de l'état 9 par épisode")
plt.xlabel("Épisode (Temps)")
plt.ylabel("Valeur")
plt.xticks(range(0,episodes+1,episodes//10))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

plt.subplot(2,2,3) # Ce graphique a été fait par l'IA

# Valeur des états
v_values = np.max(Q, axis=1).reshape(4,4)

# Affichage du fond coloré
plt.imshow(v_values, cmap="plasma", aspect="equal")
plt.colorbar(label="Valeur de l'état V(s)")

# Correspondance actions -> directions
dx_map = {0: -1, 1: 0, 2: 1, 3: 0}   # gauche, bas, droite, haut
dy_map = {0: 0, 1: 1, 2: 0, 3: -1}

# États terminaux
trous_et_but = [5, 7, 11, 12, 15]

# Parcours de tous les états
for state in range(16):

    row = state // 4
    col = state % 4

    # Trous et but
    if state in trous_et_but:

        if state == 15:
            plt.text(
                col, row,
                "BUT",
                ha="center",
                va="center",
                color="green",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none")
            )
        else:
            plt.text(
                col, row,
                "TROU",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="none")
            )

        continue

    # Actions optimales (gestion des égalités)
    q_state = Q[state]
    max_q = np.max(q_state)

    best_actions = np.where(
        np.isclose(q_state, max_q)
    )[0]

    # Une flèche par action optimale
    for action in best_actions:

        dx = dx_map[action]
        dy = dy_map[action]

        plt.quiver(
            col,
            row,
            dx,
            dy,
            angles='xy',
            scale_units='xy',
            scale=2,
            pivot='middle',
            color='grey'
        )

plt.title("Politique optimale apprise")

plt.xticks(range(4))
plt.yticks(range(4))

plt.xlabel("Colonnes")
plt.ylabel("Lignes")

plt.grid(True, linestyle="--", alpha=0.3)

plt.subplot(2,2,4) # graphique de l'évolution du score
plt.plot(scores, lw=3)
plt.xlabel("épisodes")
plt.ylabel("Score")
plt.xticks(range(0,episodes+1,episodes//10))
plt.title("évolution du score totale")

plt.grid(True, linestyle="--", alpha=0.3)


plt.tight_layout()
plt.show()