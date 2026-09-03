import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt

def load_Q(name): # chargement
    try:
        with open(f'{name}.pkl', 'rb') as file:
            return pickle.load(file)
    except:
            return {}
def dump_Q(name, Q): # déchargement
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(Q, file)


def train(episodes=int(1e6)):
    Q = {}
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999995
    epsilon_min = 0.01

    hist_lenght_Q = [0]
    hist_epsilon = [1]
    hist_victoryX = [] # joueur 1
    hist_victoryO = [] # joueur -1
    hist_egality = [] # égalités

    def check_win(state: np.ndarray, player: int):
        grid = state.reshape(3, 3)
        if np.any(np.all(grid == player, axis=0)) or np.any(np.all(grid == player, axis=1)):
            return True
        if np.all(np.diagonal(grid) == player) or np.all(np.diagonal(np.fliplr(grid)) == player):
            return True
        return False

    for episode in tqdm.tqdm(range(episodes)):
        current_state = np.zeros(9)
        current_player = 1
        done = False

        while not done:
            legal_moves = [i for i, case in enumerate(current_state) if case == 0]
            state_key = tuple(current_player * current_state)

            if state_key not in Q:
                Q[state_key] = np.zeros(9)

            # Exploration vs Exploitation
            if np.random.random() < epsilon:
                action = np.random.choice(legal_moves)
            else:
                q_values = Q[state_key]
                action = legal_moves[np.argmax(q_values[legal_moves])]

            # Application du mouvement
            next_state = current_state.copy()
            next_state[action] = current_player

            # Vérification de l'état du jeu
            if check_win(next_state, current_player):
                reward = 1
                done = True
                # Le coup gagnant reçoit +1
                Q[state_key][action] += alpha * (reward - Q[state_key][action])

                if current_player == 1:
                    hist_victoryX.append(1)
                    hist_victoryO.append(0)
                else:
                    hist_victoryO.append(1)
                    hist_victoryX.append(0)
                hist_egality.append(0)
            elif np.count_nonzero(next_state) == 9:
                reward = 0
                done = True
                # Match nul reçoit 0
                Q[state_key][action] += alpha * (reward - Q[state_key][action])

                hist_egality.append(1)
                hist_victoryX.append(0)
                hist_victoryO.append(0)

            else:
                # Le jeu continue : calcul du coup suivant du point de vue adverse
                next_key = tuple((-current_player) * next_state)
                if next_key not in Q:
                    Q[next_key] = np.zeros(9)
                
                legal_next = [i for i, case in enumerate(next_state) if case == 0]
                
                # IMPORTANT : Le gain de l'adversaire est notre perte. 
                # On soustrait donc la valeur max de l'état suivant.
                target = 0 - gamma * np.max(Q[next_key][legal_next])
                Q[state_key][action] += alpha * (target - Q[state_key][action])

            # Transition vers le joueur suivant
            current_state = next_state
            current_player *= -1

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        hist_lenght_Q.append(len(Q))
        hist_epsilon.append(epsilon)

    print(f"Taille de la table Q : {len(Q)}")
    dump_Q('MorpionTDbis', Q)

    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1) # graphique contenant la courbe epsilon et l'évolution du pourcentage des résultats

    # Afficher epsilon
    plt.plot(hist_epsilon, lw=2, label="Courbe d'epsilon", c='g')
    plt.xlabel("épisodes")
    plt.ylabel("Valeur d'epsilon", c='g')
    plt.tick_params(axis='y', labelcolor='g')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')

    plt.twinx()

    # Afficher l'évolution des pourcentages des résultats
    array = np.array([hist_victoryX, hist_egality, hist_victoryO])
    running_counts = np.cumsum(array, 1)
    counts_per_episode = np.arange(1,array.shape[1]+1)
    running_percentages = (running_counts / counts_per_episode[None, :]) * 100

    plt_hist_victoryX = running_percentages[0]
    plt_hist_egality = running_percentages[1]
    plt_hist_victoryO = running_percentages[2]

    plt.plot(plt_hist_victoryX, label="Victoires X (%)", c='black', ls='--')
    plt.plot(plt_hist_victoryO, label="Victoires O (%)", c='pink', ls='--')
    plt.plot(plt_hist_egality, label="Egalités (%)", c='cyan', lw=3)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.ylabel("Distribution des résultats (%)", c='b')
    plt.tick_params(axis='y', labelcolor='b')

    plt.legend()

    plt.subplot(1,2,2) # graphique contenant l'historique du nombre d'états explorés
    plt.plot(hist_lenght_Q, c='r', lw=3, label="Longueur de Q")
    plt.axhline(y=5478, c='b', ls='--', label="Nombre d'états totals")
    plt.axhline(y=4500, c='orange', ls='--', label="y = 4500")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xlabel("épisodes")
    plt.ylabel("nombre d'états")

    plt.tight_layout()
    plt.show()


def playAI(games=3, Q={}):
    def check_win(state, player):
        grid = state.reshape(3, 3)

        if np.any(np.all(grid == player, axis=0)):
            return True

        if np.any(np.all(grid == player, axis=1)):
            return True

        if np.all(np.diagonal(grid) == player):
            return True

        if np.all(np.diagonal(np.fliplr(grid)) == player):
            return True

        return False


    def display_board(state):
        symbols = {
            1: "X",
            -1: "O",
            0: "."
        }

        grid = [symbols[int(x)] for x in state]

        print()
        for i in range(3):
            print(" ".join(grid[3*i:3*i+3]))
        print()


    def ai_move(state, current_player):
        legal_moves = np.where(state == 0)[0]

        state_key = tuple(current_player * state) 

        if state_key not in Q:
            return np.random.choice(legal_moves)

        q_values = Q[state_key]

        return legal_moves[np.argmax(q_values[legal_moves])]


    def play_game():

        board = np.zeros(9)

        ai_player = np.random.choice([-1, 1])
        human_player = -ai_player

        print(f"L'IA joue {'X' if ai_player == 1 else 'O'}")
        print(f"Tu joues {'X' if human_player == 1 else 'O'}")

        current_player = 1

        while True:

            display_board(board)

            legal_moves = np.where(board == 0)[0]

            if current_player == human_player:

                while True:
                    try:
                        move = int(input("Case (0-8) : "))

                        if move in legal_moves:
                            break

                        print("Case invalide.")

                    except:
                        print("Entrée invalide.")

            else:

                move = ai_move(board, current_player)
                print(f"IA joue : {move}")

            board[move] = current_player

            if check_win(board, current_player):

                display_board(board)

                if current_player == human_player:
                    print("Tu as gagné !")
                else:
                    print("L'IA a gagné !")

                break

            if np.count_nonzero(board) == 9:

                display_board(board)
                print("Match nul.")
                break

            current_player *= -1


    for game in range(games):
        play_game()

train()
Q = load_Q('MorpionTDbis')
playAI(5, Q=Q)