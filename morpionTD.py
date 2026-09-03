
# agent morpion
import numpy as np , matplotlib.pyplot as plt, tqdm, time, pickle
#np.random.seed(22)

class Agent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon_decay=0.99999, is_computer=True, is_training=True,
                 player_number=1, name='1'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.is_computer = is_computer
        self.is_training = is_training

        self.player_number = player_number
        
        # ✅ Dictionnaire pour Q-values
        self.Q = {}
        """if not self.is_training:
            self.Q = self.load_Q(f'QL_{name}_50000')"""
        
        self.results = []
        self.results_random = []

    def load_Q(self, name): # chargement
        try:
            with open(f'{name}.pkl', 'rb') as file:
                return pickle.load(file)
        except:
                return {}
    def dump_Q(self, name): # déchargement
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(self.Q, file)
    
    def state_to_key(self, grid):
        """Convertit une grille en clé hashable"""
        return tuple(grid.flatten())
    
    def get_q_value(self, state_key, action):
        """Récupère une Q-value avec valeur par défaut 0"""
        return self.Q.get((state_key, action), 0.0)
    
    def legal_moves(self, grid):
        """Retourne les actions possibles"""
        moves = []
        for i in range(3):
            for j in range(3):
                if grid[i, j] == 0:
                    moves.append(i * 3 + j)
        return moves
    
    def epsilon_greedy(self, grid):
        state_key = self.state_to_key(grid)
        moves = self.legal_moves(grid)
        
        self.epsilon = max(self.epsilon*self.epsilon_decay, 0.0005)
        if np.random.random() < self.epsilon:
            #print(moves)
            return np.random.choice(moves)
        else:
            # Trouver la meilleure action parmi les coups légaux
            q_values = [self.get_q_value(state_key, a) for a in moves]
            return moves[np.argmax(q_values)]
    
    def eval(self, grid):
        # Pour le joueur courant (self.player_number)
        if check_win(self.player_number, grid):
            return 10
        # Pour l'adversaire (l'autre joueur)
        if check_win(-self.player_number, grid):
            return -10
        return 0
    
    def Q_learning(self, grid, action, other_player):
        # État courant
        state_key = self.state_to_key(grid)

        # Prochain état (copie!)
        sp = grid.copy()
        i, j = action // 3, action % 3
        sp[i, j] = self.player_number  # ou -1 selon le joueur
        if not( check_win(self.player_number, sp) or check_draw(sp)):
            opponenent_action = other_player.epsilon_greedy(sp)
            i, j = opponenent_action // 3, opponenent_action % 3
            sp[i, j] = -self.player_number
        sp_key = self.state_to_key(sp)
        
        # Récompense
        R = self.eval(sp)
        
        # Q-value actuelle
        current_q = self.get_q_value(state_key, action)
        
        # Q-value future (max des actions légales)
        future_moves = self.legal_moves(sp)
        if future_moves and R == 0:  # Si partie continue
            future_q = max([self.get_q_value(sp_key, a) for a in future_moves], default=0)
        else:
            future_q = 0  # Partie terminée
        
        # Mise à jour
        new_q = current_q + self.alpha * (R + self.gamma * future_q - current_q)
        self.Q[(state_key, action)] = new_q
    
    def play(self, grid, other_player):
        moves = self.legal_moves(grid)
        
        if self.is_computer:
            if self.is_training:
                action = self.epsilon_greedy(grid)
                self.Q_learning(grid, action, other_player)
                return action
            else:
                # Exploitation pure
                state_key = self.state_to_key(grid)
                q_values = [self.get_q_value(state_key, a) for a in moves]
                return moves[np.argmax(q_values)]
        else:
            action = int(input("$> "))
            return action if action in moves else moves[0]
     


def check_win(player:int, grid:np.ndarray):
    for i in range(3):
        if np.all(grid[i, :] == player): return True

        
    for j in range(3):
        if np.all(grid[:, j] == player): return True
    
    if np.all(np.diag(grid) == player): return True
    
    if np.all(np.diag(np.fliplr(grid)) == player): return True

    return False

def check_draw(grid:np.ndarray):
    if 0 in grid.ravel():
        return False
    return True

p1 = Agent()
p2 = Agent(player_number=-1, name='2')
p4 = Agent(epsilon_decay=1, player_number=-1)

def game(p1:Agent,p2:Agent, current_step:int, random_present = False):
    players = [p1,p2]
    np.random.shuffle(players)
    grid = np.zeros((3,3))

    is_game_finished = False
    while not is_game_finished:
        for i,player in enumerate(players):
            if not player.is_computer :
                print(grid)
            action = player.play(grid, players[1-i])
            grid[action//3][action%3] = player.player_number
            if check_win(player.player_number, grid):
                is_game_finished = True
                if not random_present:
                    player.results.append(1)
                    players[1-i].results.append(-1)
                else:
                    player.results_random.append(1)
                    players[1-i].results_random.append(-1)
                break

            if check_draw(grid):
                is_game_finished = True
                if not random_present:
                    
                    player.results.append(0)
                    players[1-i].results.append(0)
                else:

                    player.results_random.append(0)
                    players[1-i].results_random.append(0)
                break

    
def run():
    n_steps = int(5e4)
    timer = time.time()
    for step in tqdm.tqdm(range(n_steps)):
        p1.is_training = True
        p2.is_training = True
        game(p1,p2, step)

        p1.is_training = False
        p2.is_training = False
        p4.player_number = -1
        game(p1,p4,2,random_present=True)
        p4.player_number = 1
        game(p2,p4,2,random_present=True)
        

    p1.dump_Q('QL_1_50000')
    p2.dump_Q('QL_2_50000')

    intervall = time.time() - timer
    print(intervall)


def run_human():
    human = Agent(player_number=-1, is_computer=False)
    p1.is_training = False
    p1.Q = p1.load_Q('QL_1_50000')
    print("Vous jouez contre l'agent 1")
    game(p1,human, 2)
    print("Vous jouez contre l'agent 2")
    p2.is_training = False
    p2.Q = p1.load_Q('QL_2_50000')
    human.player_number = 1
    game(p2,human,2)



def show_graphics():
    plt.figure(figsize=(20,8))
    plt.suptitle("Résultats sur 50000 itérations d'un agent apprenant à jouer au morpion grâce au Q-learning", fontsize=20)
    plt.subplot(2,3,1)
    plt.title("Agent 1 Vs Agent 2 (global)")
    names = ['Agent 1', 'Agent 2', 'matchs nuls']
    values = [p1.results.count(1)/len(p1.results)*100,
            p2.results.count(1)/len(p2.results)*100,p1.results.count(0)/len(p1.results)*100]
    plt.ylabel('Pourcentages')
    plt.bar(names,values, color=['green','red','black'])

    plt.subplot(2,3,2)
    grid_init = tuple(np.zeros(9, dtype=int))
    q_values_p1 = [p1.Q.get((grid_init, a), 0) for a in range(9)]
    q_values_p2 = [p2.Q.get((grid_init, a), 0) for a in range(9)]


    plt.title("Q-table de l'état initial de l'agent 1")
    plt.bar(range(9),q_values_p1, color='green')

    plt.subplot(2,3,3)
    plt.title("Q-table de l'état initial de l'agent 2")
    plt.bar(range(9), q_values_p2, color='red')

    plt.subplot(2,3,4)
    plt.title("Agent 1 Vs Agent aléatoire (global)")
    names = ['Agent 1', 'Agent random', 'matchs nuls']
    values = [p1.results_random.count(1)/len(p1.results_random)*100,
            p1.results_random.count(-1)/len(p1.results_random)*100,p1.results_random.count(0)/len(p1.results_random)*100]
    plt.ylabel('Pourcentages')
    plt.bar(names,values, color=['green','blue','black'])

    plt.subplot(2,3,5)
    plt.title("Evolution du nombre de victoire au fil du temps agent 1 vs agent random")
    values = [0]
    for result in p1.results_random:
        if result == 1:
            values.append(values[-1]+1)
        else:
            values.append(values[-1])
    plt.plot(values)

    plt.subplot(2,3,6)
    plt.title("Agent 2 Vs Agent aléatoire (global)")
    names = ['Agent 1', 'Agent random', 'matchs nuls']
    values = [p2.results_random.count(1)/len(p2.results_random)*100,
            p2.results_random.count(-1)/len(p2.results_random)*100,p2.results_random.count(0)/len(p2.results_random)*100]
    plt.ylabel('Pourcentages')
    plt.bar(names,values, color=['red','blue','black'])


    plt.show()

def train():
    run()
    show_graphics()

train()
run_human()

"""
output après training:

-> imbattable > (j'ai pas réussi)

"""