"""
L'environnement içi sera composé d'une liste de 9 cases. L'agent sera à la case 4 au début. Chaque case donnera -1,
 sauf la case 0 -> -2 et le 8 -> 1
"""

import numpy as np
import matplotlib.pyplot as plt

rewards = [-2,-1,-1,-1,-1,-1,-1,-1,1]
actions = [-1,1]
values = np.zeros(len(rewards))
print(values)


epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.05
gamma = 0.9

# pour les résultats
epsilons = []

colors_hist =  [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3",
        "#33FFF0", "#FFA500", "#800080", "#008080", "#FFC0CB",
        "#A52A2A", "#808080", "#00FFFF", "#FFD700", "#4B0082",
        "#00FF00", "#FF4500", "#2E8B57", "#DA70D6", "#1E90FF"
    ]

def TD():
    global epsilon, values, current_pos
    for i in range(100):
        score = 0
        done = False
        current_pos = 4
        while not done:
            if epsilon > np.random.random(): # explorer
                a = np.random.choice(actions)
            else: # exploiter
                a = np.argmax([values[current_pos-1], values[current_pos+1]]) * 2 -1 # Pour avoir -1 ou 1
            current_pos += a
            r = rewards[current_pos]
            if r != -1: # état terminal
                done = True
            print(r)
            # actualiser V
            values[current_pos-a] = r + gamma*values[current_pos]

            epsilon = max(epsilon_min, epsilon_decay*epsilon)
            
        # mettre à jour l'historique
        epsilons.append(epsilon)

   
    print(values)

    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    plt.title(r"Evolution d'epsilon ($\epsilon$) en fonction du temps")
    plt.plot(epsilons, c='r', label="valeur d' epsilon")
    plt.axhline(epsilon_min, color='black', linewidth=1, linestyle='--', label="valeur minimale d'epsilon")
    plt.xlabel('épisode')
    plt.ylabel("valeur d'epsilon")
    plt.legend()

    plt.subplot(1,2,2) # L'IA m'a aidé pour ce graphique

    states = [f"État {i}" for i in range(len(rewards))]

    bars = plt.bar(states, values, color=colors_hist, edgecolor='black', width=0.6)

    plt.title("Valeur finale estimée pour chaque état (V)", fontsize=14, )
    plt.xlabel("États de l'environnement", fontsize=12)
    plt.ylabel("Valeur (V)", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--') # Ligne de repère à 0
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        # Ajuste la position du texte selon que la valeur est positive ou négative
        va_dir = 'bottom' if yval >= 0 else 'top'
        offset = 0.05 if yval >= 0 else -0.05
        
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + offset, 
                f"{yval:.2f}", ha='center', va=va_dir, fontweight='bold')
    plt.text(5, -3, r"$v_{\pi}(s) = R_{t+1} + \gamma v_{\pi}(s')$", fontsize=12, fontweight='bold')

    plt.show()

def DP():
    global values
    hist = [values.copy()]
    for episode in range(8):
        for i in range(1, len(rewards)-1): # Ne pas traiter les états terminaux
            state_left = i - 1
            r_left = rewards[state_left]
            v_left = r_left + gamma*values[state_left]

            state_right = i +1
            r_right = rewards[state_right]
            v_right = r_right + gamma*values[state_right]

            values[i] = max(v_left, v_right)
            
        hist.append(values.copy())
    print(values)

    plt.figure(figsize=(10,5))
    for s in range(len(rewards)):
        L = []
        for l in hist:
            L.append(l[s])
        plt.plot(L, c=colors_hist[s], label=f"valeur de l'état {s}")

    plt.xlabel("épisodes")
    plt.ylabel("valeur d'état")
    plt.legend()
    plt.title("Figure gridworld 1D DP")
    plt.show()
TD()

