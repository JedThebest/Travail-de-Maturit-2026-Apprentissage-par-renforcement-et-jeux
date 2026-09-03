
import numpy as np
import matplotlib.pyplot as plt

states = {}

initial_state = np.zeros(9)
print(type(initial_state))

states[tuple(initial_state)] = 0

def check_in_states(state:np.ndarray):
    return tuple(state) in states

def check_win(state:np.ndarray, player:int):
    grid = state.reshape(3, 3)
    # Vérification des lignes et des colonnes
    if np.any(np.all(grid == player, axis=0)) or np.any(np.all(grid == player, axis=1)):
        return True
    # Vérification des deux diagonales
    if np.all(np.diagonal(grid) == player) or np.all(np.diagonal(np.fliplr(grid)) == player):
        return True
    return False

def get_all_states(state:np.ndarray):
    global states
    # On part d'un état et ça retourne tous les états possibles

    current_player = 1 if np.count_nonzero(state) % 2 == 0 else -1

    if not check_win(state, -current_player):
        for i,case in enumerate(state):
            if case == 0:
                new_state = state.copy()
                new_state[i] = current_player
                if not check_in_states(new_state):
                    states[tuple(new_state)] = 0
                    get_all_states(new_state)


get_all_states(initial_state)
print(len(states))


values = {}
for state in states:
    values[state] = 0

theta = 1e-6
c = 0


while True:
    c += 1
    delta = 0
    prep_values = values.copy()
    for state in states:
        current_player = 1 if np.count_nonzero(state) % 2 == 0 else -1

        if check_win(np.array(state), -current_player):
            values[state] = -current_player
            continue
        elif np.count_nonzero(state) == 9:
            values[state] = 0
            continue

        v_sp = []
        for i,case in enumerate(state):
            if case == 0:
                new_state = list(state)
                new_state[i] = current_player
                new_state_tuple = tuple(new_state)

                 # Vérification immédiate si ce coup est gagnant
                if check_win(np.array(new_state), current_player):
                    v_sp.append(current_player)
                elif new_state_tuple in values:
                    v_sp.append(values[new_state_tuple])

        if current_player == 1:
            # Le joueur 1 cherche à maximiser le score
            values[state] = max(v_sp)
        else:
            # Le joueur -1 cherche à minimiser le score
            values[state] = min(v_sp)

        delta = max(delta, abs(prep_values[state] - values[state]))
    if delta < theta:break

print(c)

# Jouer avec l'humain IA made it
def afficher_plateau(state_tuple):
    """Affiche le plateau de morpion de manière lisible."""
    symboles = {0: " ", 1: "X", -1: "O"}
    grid = [symboles[x] for x in state_tuple]
    print("\n   0   1   2")
    print(f"0  {grid[0]} | {grid[1]} | {grid[2]}")
    print("  ---+---+---")
    print(f"1  {grid[3]} | {grid[4]} | {grid[5]}")
    print("  ---+---+---")
    print(f"2  {grid[6]} | {grid[7]} | {grid[8]}\n")

def jouer_partie():
    # L'état initial est un tuple de 9 zéros
    board = np.zeros(9)
    print("Bienvenue dans le Morpion IA !")
    print("Vous jouez avec les 'O' (Joueur -1). L'IA joue avec les 'X' (Joueur 1).")
    agent_player = np.random.choice([-1,1])
    current_player = 1
    
    while True:
        if current_player == agent_player:

            legal_moves = [i for i,case in enumerate(board) if board[i] == 0]

            score = -float('inf') if agent_player == 1 else float('inf')
            best_move = None

            for move in legal_moves:
                s = board.copy()
                s[move] = agent_player
                if agent_player == 1:
                    if values[tuple(s)] > score:
                        score = values[tuple(s)]
                        best_move = move
                else:
                    if values[tuple(s)] < score:
                        score = values[tuple(s)]
                        best_move = move
            
            # jouer best_move
            board[best_move] = agent_player

            if check_win(board, agent_player):
                print("L'IA a gagné !")
                break
            elif np.count_nonzero(board) == 9:
                print("égalités")
                break
            else:
                current_player *= -1

        else:
            afficher_plateau(board)
            valid_move = False
            while not valid_move:
                try:
                    case = int(input("Choisissez une case vide (0 à 8) : "))
                    if 0 <= case <= 8 and board[case] == 0:
                        valid_move = True
                    else:
                        print("Case invalide ou déjà occupée. Réessayez.")
                except ValueError:
                    print("Veuillez entrer un nombre entre 0 et 8.")
                    
            board[case] = current_player


            if check_win(board, current_player):
                print("L'humain a gagné !")
                break
            elif np.count_nonzero(board) == 9:
                print("égalités")
                break
            else:
                current_player *= -1


# Lancement de la partie
jouer_partie()

# L'IA a fait ces graphiques

nb_victoires_X = sum(1 for v in values.values() if v == 1.0)
nb_victoires_O = sum(1 for v in values.values() if v == -1.0)
nb_matchs_nuls = sum(1 for v in values.values() if v == 0.0)

labels = ["Victoire X ", "Match Nul ", "Victoire O "]
sizes = [nb_victoires_X, nb_matchs_nuls, nb_victoires_O]
colors = ["#ff9999", "#66b3ff", "#99ff99"]
explode = (0.05, 0.05, 0.05)  # Espace subtil entre les parts

plt.figure(figsize=(12, 5))

# Création du sous-graphique 1 (à gauche)
plt.subplot(1, 2, 1)
plt.pie(
    sizes, 
    explode=explode, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%', 
    startangle=140, 
    shadow=True
)
plt.title("Répartition de la valeur des états uniques")

# =====================================================================
# GRAPHIQUE 3 : Nombre d'états selon la profondeur (Histogramme)
# =====================================================================

# Dictionnaire pour compter les états par nombre de pions posés
profondeurs = {i: 0 for i in range(10)}

for state in states:
    # Le nombre de pions posés correspond au nombre de cases non vides (différentes de 0)
    nb_pions = np.count_nonzero(state)
    profondeurs[nb_pions] += 1

x_bars = list(profondeurs.keys())
y_bars = list(profondeurs.values())

# Création du sous-graphique 2 (à droite)
plt.subplot(1, 2, 2)
bars = plt.bar(x_bars, y_bars, color="#bcbd22", edgecolor="black", alpha=0.8)

# Ajouter les valeurs exactes au-dessus de chaque barre
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            yval + 30, 
            str(yval), 
            ha='center', 
            va='bottom', 
            fontsize=9
        )

plt.title("Nombre d'états selon le nombre de pions posés")
plt.xlabel("Nombre de pions sur le plateau (Profondeur)")
plt.ylabel("Nombre d'états uniques détectés")
plt.xticks(range(10))
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Ajuster l'espacement et afficher les graphiques
plt.tight_layout()
plt.show()


