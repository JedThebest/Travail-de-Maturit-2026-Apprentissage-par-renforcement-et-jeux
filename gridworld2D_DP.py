
import numpy as np
import matplotlib.pyplot as plt

env = np.zeros(15) # représente une grille 5x3
print(env)
actions = np.arange(4) # 0 <- 1 monter 2 -> 3 descendre
print(actions)

values = np.zeros(15)

colors_hist =  [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3",
        "#33FFF0", "#FFA500", "#800080", "#008080", "#FFC0CB",
        "#A52A2A", "#808080", "#00FFFF", "#FFD700", "#4B0082",
        "#00FF00", "#FF4500", "#2E8B57", "#DA70D6", "#1E90FF"
    ]

def reward(pos):
    if pos == 0 :
        r = -10
    elif pos == len(env)-1:
        r = 1
    else:
        r = -1
    return r

def move(pos:int, action:int):
    if action == 0:
        if pos % 5 != 0:
            return pos-1
    elif action == 1:
        if pos > 5:
            return pos -5
    elif action == 2:
        if pos % 5 != 4:
            return pos +1
    else:
        if pos < 10:
            return pos +5
        
    return pos
gamma = 0.9
historic = [values.copy()]
for episode in range(10):
    for pos in range(1,len(env)-1):
        v_sp = []
        for action in actions:
            sp = move(pos,action)
            r = reward(sp)
            v = r + gamma*values[sp]
            v_sp.append(v)
        values[pos] = max(v_sp)
    historic.append(values.copy())

print(values)

plt.figure(figsize=(15,5))

historic = np.array(historic)
plt.subplot(1,2,1)
for s in range(len(env)):
    plt.plot(historic[:,s], c= colors_hist[s], label=f"valeur d'état {s}")
plt.title("Évolution des valeurs d'états par épisode")
plt.xlabel("Épisode (Temps)")
plt.ylabel("Valeur")
plt.xticks(range(11))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

plt.subplot(1,2,2) # L'IA m'a aidé pour ce graphique

# Redimensionnement du vecteur en matrice 3 lignes x 5 colonnes
grid_values = values.reshape(3, 5)

plt.imshow(grid_values, cmap='plasma', aspect='equal')
plt.title("Heatmap finale des valeurs (Grille 5x3)")

# Ajout des valeurs numériques dans chaque case
for i in range(3):
    for j in range(5):
        val = grid_values[i, j]
        text_color = "white" if val < -1.5 else "black"
        # plt.text(x, y, texte, ...)
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
        # placer l'index de l'état
        n = j + 5*i
        plt.text(j+0.4, i+0.4, f"{n}", ha="center", va="center", color=text_color, fontsize=8)

# Configuration des axes et de la barre de couleur
plt.xticks(range(5))
plt.yticks(range(3))
plt.xlabel("Colonnes")
plt.ylabel("Lignes")
plt.colorbar(label="Valeur de l'état")


plt.tight_layout()
plt.show()