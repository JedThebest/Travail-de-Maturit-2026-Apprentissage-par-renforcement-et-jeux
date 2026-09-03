import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped

values = np.zeros(env.observation_space.n)

gamma = 0.9

theta = 1e-6

c = 0

historic = [values.copy()]

while True:
    delta = 0
    c += 1
    prep_values = values.copy()
    for state in range(env.observation_space.n):
        v_sp = []
        for action in range(env.action_space.n):
            v = 0
            for prob, sp, r, _ in env.P[state][action]:
                v += prob*( r + gamma*prep_values[sp] )
            v_sp.append(v)
        values[state] = max(v_sp)

        delta = max(delta, abs(prep_values[state] - values[state]))
    historic.append(values.copy())
    if delta < theta:break

print(values, c)

# extraction de la politique
policy = np.zeros(env.observation_space.n)
for state in range(len(policy)):
    
    action_values = np.zeros(env.action_space.n)

    for action in range(env.action_space.n):
        for prob, sp, r, _ in env.P[state][action]:
            action_values[action] += prob*( r + gamma*prep_values[sp] )

    policy[state] = np.argmax(action_values)

print(policy)
colors_hist = plt.cm.jet(np.linspace(0, 1, env.observation_space.n + 1))
plt.figure(figsize=(14,5))

historic = np.array(historic)
plt.subplot(1,2,1)
for s in range(env.observation_space.n):
    plt.plot(historic[:,s], c= colors_hist[s], label=f"valeur d'état {s}")
plt.title("Évolution des valeurs d'états par épisode")
plt.xlabel("Épisode (Temps)")
plt.ylabel("Valeur")
plt.xticks(range(0,c+1,5) if c > 10 else range(c+1))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)

plt.subplot(1,2,2) # L'IA m'a aidé pour ce graphique

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


plt.subplots_adjust(wspace=0.1, bottom=0.3)
plt.show()

# GRAPHIQUE 100% fait par l'IA
# --- AJOUT DE LA NOUVELLE FIGURE : POLITIQUE OPTIMALE APPRISE ---

# 1. Configuration des directions FrozenLake : 0:Gauche, 1:Bas, 2:Droite, 3:Haut
actions_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# Redimensionnement de la politique et des valeurs pour la grille 4x4
grid_policy = policy.reshape(4, 4)
grid_values = values.reshape(4, 4)

# Définition des états spéciaux (Trous et But standard de FrozenLake 4x4)
holes = [5, 7, 11, 12]
goal = 15

plt.figure(figsize=(6, 5))

# Affichage du fond coloré basé sur la valeur des états
plt.imshow(grid_values, cmap="plasma", aspect="equal")
plt.title("Politique optimale apprise", fontweight="bold", pad=15)

# Remplissage de la grille avec les flèches et les textes
for i in range(4):
    for j in range(4):
        state_idx = i * 4 + j
        val = grid_values[i, j]
        text_color = "white" if val < 0.4 else "black"

        if state_idx in holes:
            # Case Trou
            plt.text(
                j,
                i,
                "TROU",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
        elif state_idx == goal:
            # Case But
            plt.text(
                j,
                i,
                "BUT",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            )
        else:
            # Case normale : Afficher la flèche de l'action optimale
            action_idx = int(grid_policy[i, j])
            symbol = actions_symbols[action_idx]
            plt.text(
                j,
                i,
                symbol,
                ha="center",
                va="center",
                color=text_color,
                fontsize=18,
                fontweight="bold",
            )

# Configuration esthétique des axes
plt.xticks(range(4))
plt.yticks(range(4))
plt.xlabel("Colonnes")
plt.ylabel("Lignes")

# Ajout de la barre de couleur latérale identique à votre exemple
plt.colorbar(label="Valeur de l'état V(s)")

plt.tight_layout()
plt.show()


"""
Synthèse des résultats brièvement:
Le modèle triche. 
"""