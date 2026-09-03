import numpy as np
import matplotlib.pyplot as plt

MAX_STICKS = 14
action_space = [1,2,3]

def reward(sticks):
    if sticks <= 0:
        return -1
    if sticks == 1:
        return 1
    else:
        return 0

values = np.zeros(MAX_STICKS+1)
historic = [values.copy()]
gamma = 0.9
theta = 1e-3
delta = float('inf')
c = 0

while delta > theta:
    delta = 0
    c+= 1
    for state in range(2, MAX_STICKS+1):
        u = values[state]
        v_sp = []
        for action in action_space:
            sp = state - action
            if sp >= 0:
                v_sp.append(-(reward(sp) + gamma*values[sp]))
        if v_sp: 
            values[state] = max(v_sp)
        delta = max(delta, abs(u - values[state]))
    historic.append(values.copy())

print("Valeurs :", values)

# Extraction de la politique
policy = np.zeros(MAX_STICKS+1)
for state in range(2, MAX_STICKS + 1):
    a = 0
    bv = float("inf")
    for action in action_space:
        if state - action >= 0:
            if bv > values[state-action]:
                bv = values[state-action]
                a = action
    policy[state] = a

print("Politique de l'IA (nombre de bâtonnets à prendre) :")
print(policy)

# Joueur avec l'humain
def play():
    sticks = MAX_STICKS
    done = False
    current_player = np.random.choice([0,1])
    while not done:
        if current_player:
            a = int(input(f"Il reste {sticks} batons, combien voulez-vous en prendre ? "))
        else:
            a = 0
            bv = float("inf")
            for action in action_space:
                if sticks - action >= 0:
                    if bv > values[sticks-action]:
                        bv = values[sticks-action]
                        a = action

        sticks -= a
        if sticks <= 0:
            print(f"Nous avons un vainqueur : {'L\'IA' if current_player else 'L\'humain'}")
            done = True
        current_player = 1 - current_player

# Décommentez la ligne ci-dessous si vous souhaitez jouer dans votre terminal
# play()

# Palette de couleurs pour l'historique
colors_hist = plt.cm.jet(np.linspace(0, 1, MAX_STICKS + 1))

plt.figure(figsize=(12, 8))

gs = plt.GridSpec(2, 2)

# --- 1ER GRAPHIQUE ---
plt.subplot(gs[0, 0])

historic = np.array(historic)

for s in range(MAX_STICKS + 1):
    plt.plot(historic[:, s], c=colors_hist[s], label=f"État {s}")

plt.title("Évolution des valeurs au fil des itérations")
plt.xlabel("Itération")
plt.ylabel("Valeur de l'état")
plt.xticks(range(c + 1))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    fontsize=9
)


# --- 2ÈME GRAPHIQUE ---
plt.subplot(gs[0, 1])

grid_values = values.reshape(1, MAX_STICKS + 1)

plt.imshow(
    grid_values,
    cmap='plasma',
    aspect='auto'
)

plt.title("Valeurs finales par nombre de bâtonnets")

for j in range(MAX_STICKS + 1):

    val = grid_values[0, j]
    text_color = "black" if val > 0 else "white"

    plt.text(
        j,
        0,
        f"{val:.1f}",
        ha="center",
        va="center",
        color=text_color,
        fontweight='bold'
    )

    plt.text(
        j,
        0.35,
        f"{j}",
        ha="center",
        va="center",
        color=text_color,
        fontsize=9
    )

plt.xticks(range(MAX_STICKS + 1))
plt.yticks([])

plt.xlabel("Nombre de bâtonnets restants")

plt.colorbar(
    label="Valeur (Gagnant +1 / Perdant -1)",
    orientation='horizontal',
    pad=0.15
)


# --- 3ÈME GRAPHIQUE : CENTRÉ ---
plt.subplot(gs[1, :])

states_indices = np.arange(2, MAX_STICKS + 1)

bar_colors = [
    'darkgray' if a == 0 else plt.cm.viridis(a / 3)
    for a in policy[2:]
]

bars = plt.bar(
    states_indices,
    policy[2:],
    color=bar_colors,
    edgecolor='black',
    alpha=0.85
)

for bar in bars:

    height = bar.get_height()

    if height > 0:

        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.05,
            f'-{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
            color='black'
        )

plt.title("Politique Optimale : Coup à jouer")
plt.xlabel("Nombre de bâtonnets restants (État)")
plt.ylabel("Bâtonnets à retirer (Action)")
plt.xticks(range(MAX_STICKS + 1))
plt.yticks([0, 1, 2, 3])
plt.ylim(0, 3.5)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()