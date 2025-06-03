import numpy as np
import pickle
from collections import defaultdict
import random

class UltimateTicTacToeEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 3x3 grille de sous-grilles 3x3 - chaque élément est une grille 3x3
        self.small_boards = []
        for big_row in range(3):
            board_row = []
            for big_col in range(3):
                small_board = [[0 for _ in range(3)] for _ in range(3)]
                board_row.append(small_board)
            self.small_boards.append(board_row)
        
        self.main_board = [[0 for _ in range(3)] for _ in range(3)]
        self.active_board = None  # None signifie que tous les plateaux sont actifs
        self.last_move = None
        return self.get_state()
    
    def get_state(self):
        """Convertit l'état du jeu en tuple pour utilisation comme clé de dictionnaire"""
        # Aplatir les petites grilles
        small_flat = []
        for big_row in range(3):
            for big_col in range(3):
                for small_row in range(3):
                    for small_col in range(3):
                        small_flat.append(self.small_boards[big_row][big_col][small_row][small_col])
        
        # Aplatir la grille principale
        main_flat = []
        for row in range(3):
            for col in range(3):
                main_flat.append(self.main_board[row][col])
        
        # Ajouter l'information du plateau actif
        active_info = []
        if self.active_board is None:
            active_info = [1] * 9  # Tous actifs
        else:
            active_info = [0] * 9
            active_board_idx = self.active_board[0] * 3 + self.active_board[1]
            active_info[active_board_idx] = 1
        
        return tuple(small_flat + main_flat + active_info)
    
    def is_valid_move(self, big_row, big_col, small_row, small_col):
        # Vérifier si la case est libre
        if self.small_boards[big_row][big_col][small_row][small_col] != 0:
            return False
        
        # Vérifier si la sous-grille est déjà gagnée
        if self.main_board[big_row][big_col] != 0:
            return False
        
        # Vérifier si c'est le bon plateau actif
        if self.active_board is not None:
            return (big_row, big_col) == self.active_board
        
        return True
    
    def get_valid_actions(self):
        """Retourne la liste des actions valides"""
        valid_actions = []
        
        # Déterminer quels plateaux sont jouables
        valid_boards = []
        if self.active_board is not None:
            # Si un plateau spécifique est actif
            big_row, big_col = self.active_board
            if self.main_board[big_row][big_col] == 0:  # Plateau pas encore gagné
                valid_boards.append((big_row, big_col))
            else:
                # Si le plateau actif est gagné/plein, tous les plateaux libres sont valides
                for r in range(3):
                    for c in range(3):
                        if self.main_board[r][c] == 0:
                            valid_boards.append((r, c))
        else:
            # Tous les plateaux non-gagnés sont actifs
            for r in range(3):
                for c in range(3):
                    if self.main_board[r][c] == 0:
                        valid_boards.append((r, c))
        
        # Pour chaque plateau valide, trouver les cases libres
        for big_row, big_col in valid_boards:
            for small_row in range(3):
                for small_col in range(3):
                    if self.small_boards[big_row][big_col][small_row][small_col] == 0:
                        action = self.move_to_action(big_row, big_col, small_row, small_col)
                        valid_actions.append(action)
        
        return valid_actions
    
    def get_valid_boards(self):
        """Retourne les plateaux valides pour l'affichage"""
        valid_boards = []
        
        if self.active_board is not None:
            big_row, big_col = self.active_board
            if self.main_board[big_row][big_col] == 0:
                # Le plateau actif est encore jouable
                valid_boards.append((big_row, big_col))
            else:
                # Le plateau actif est gagné/plein, tous les plateaux libres sont valides
                for r in range(3):
                    for c in range(3):
                        if self.main_board[r][c] == 0:
                            valid_boards.append((r, c))
        else:
            # Tous les plateaux non-gagnés sont valides
            for r in range(3):
                for c in range(3):
                    if self.main_board[r][c] == 0:
                        valid_boards.append((r, c))
        
        return valid_boards
    
    def move_to_action(self, big_row, big_col, small_row, small_col):
        """Convertit un mouvement en action (entier)"""
        return big_row * 27 + big_col * 9 + small_row * 3 + small_col
    
    def action_to_move(self, action):
        """Convertit une action en mouvement"""
        big_row = action // 27
        big_col = (action % 27) // 9
        small_row = (action % 9) // 3
        small_col = action % 3
        return big_row, big_col, small_row, small_col
    
    def check_winner(self, board):
        """Vérifie le gagnant d'une grille 3x3"""
        # Lignes
        for row in board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # Colonnes
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != 0:
                return board[0][col]
        
        # Diagonales
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]
        
        return 0
    
    def check_small_board_winner(self, big_row, big_col):
        """Vérifie le gagnant d'une sous-grille"""
        return self.check_winner(self.small_boards[big_row][big_col])
    
    def is_small_board_full(self, big_row, big_col):
        """Vérifie si une sous-grille est pleine"""
        for row in range(3):
            for col in range(3):
                if self.small_boards[big_row][big_col][row][col] == 0:
                    return False
        return True
    
    def make_move(self, big_row, big_col, small_row, small_col, player):
        """Effectue un mouvement et retourne (reward, done, winner)"""
        if not self.is_valid_move(big_row, big_col, small_row, small_col):
            return -10, False, None  # Mouvement invalide
        
        # Effectuer le mouvement
        self.small_boards[big_row][big_col][small_row][small_col] = player
        self.last_move = (big_row, big_col, small_row, small_col)
        
        # Vérifier si la sous-grille est gagnée
        small_winner = self.check_small_board_winner(big_row, big_col)
        if small_winner != 0:
            self.main_board[big_row][big_col] = small_winner
        elif self.is_small_board_full(big_row, big_col):
            self.main_board[big_row][big_col] = 99  # Match nul dans cette grille
        
        # Déterminer le prochain plateau actif
        next_big_row, next_big_col = small_row, small_col
        if (self.main_board[next_big_row][next_big_col] == 0 and 
            not self.is_small_board_full(next_big_row, next_big_col)):
            self.active_board = (next_big_row, next_big_col)
        else:
            self.active_board = None  # Tous les plateaux sont actifs
        
        # Vérifier la victoire globale
        main_winner = self.check_winner(self.main_board)
        if main_winner != 0 and main_winner != 99:
            reward = 100 if main_winner == player else -100
            return reward, True, main_winner
        
        # Vérifier le match nul global
        if all(self.main_board[r][c] != 0 for r in range(3) for c in range(3)):
            return 0, True, 0  # Match nul
        
        # Récompense pour gagner une sous-grille
        if small_winner == player:
            reward = 10
        elif small_winner != 0:
            reward = -10
        else:
            reward = 1  # Petit bonus pour un mouvement valide
        
        return reward, False, None

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
    
    def get_action(self, state, epsilon=None):
        """Sélectionne une action selon la politique epsilon-greedy"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # Reconstruire l'environnement à partir de l'état
        env = UltimateTicTacToeEnv()
        env.small_boards = self.state_to_boards(state)
        
        # Reconstruire le main_board et active_board à partir de l'état
        main_board_start = 81  # Après les 81 cases des petites grilles
        idx = main_board_start
        for r in range(3):
            for c in range(3):
                env.main_board[r][c] = state[idx]
                idx += 1
        
        # Reconstruire active_board à partir des informations d'activité
        active_info_start = 90  # Après main_board
        active_info = state[active_info_start:active_info_start+9]
        
        if sum(active_info) == 9:  # Tous actifs
            env.active_board = None
        else:
            # Trouver le plateau actif
            for i, is_active in enumerate(active_info):
                if is_active == 1:
                    env.active_board = (i // 3, i % 3)
                    break
        
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        if random.random() < epsilon:
            # Exploration : action aléatoire
            return random.choice(valid_actions)
        else:
            # Exploitation : meilleure action connue
            q_values = {action: self.q_table[state][action] for action in valid_actions}
            if not q_values:
                return random.choice(valid_actions)
            
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def state_to_boards(self, state):
        """Reconstruit les plateaux à partir de l'état"""
        small_boards = []
        for big_row in range(3):
            board_row = []
            for big_col in range(3):
                small_board = [[0 for _ in range(3)] for _ in range(3)]
                board_row.append(small_board)
            small_boards.append(board_row)
        
        idx = 0
        for big_row in range(3):
            for big_col in range(3):
                for small_row in range(3):
                    for small_col in range(3):
                        small_boards[big_row][big_col][small_row][small_col] = state[idx]
                        idx += 1
        return small_boards
    
    def update_q_table(self, state, action, reward, next_state):
        """Met à jour la Q-table selon l'équation de Bellman"""
        current_q = self.q_table[state][action]
        
        if next_state is None:
            # État terminal
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            # Calculer la valeur maximale pour l'état suivant
            env = UltimateTicTacToeEnv()
            env.small_boards = self.state_to_boards(next_state)
            valid_next_actions = env.get_valid_actions()
            
            if valid_next_actions:
                max_next_q = max(self.q_table[next_state][a] for a in valid_next_actions)
            else:
                max_next_q = 0
            
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filename):
        """Sauvegarde la Q-table"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, filename):
        """Charge la Q-table"""
        try:
            with open(filename, 'rb') as f:
                q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in q_dict.items():
                    for action, value in actions.items():
                        self.q_table[state][action] = value
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Utilisation d'une Q-table vide.")

# Classe pour un agent aléatoire (utile pour l'entraînement)
class RandomAgent:
    def __init__(self):
        pass
    
    def get_action(self, state, valid_actions):
        """Retourne une action aléatoire parmi les actions valides"""
        if not valid_actions:
            return None
        return random.choice(valid_actions)

# Fonction utilitaire pour entraîner l'agent
def train_agent(agent, num_episodes=1000, opponent='random'):
    """Entraîne l'agent contre un adversaire"""
    env = UltimateTicTacToeEnv()
    wins = 0
    draws = 0
    losses = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        current_player = 1  # L'agent commence
        
        while not done:
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            if current_player == 1:
                # Tour de l'agent
                action = agent.get_action(state)
                if action is not None:
                    big_row, big_col, small_row, small_col = env.action_to_move(action)
                    reward, done, winner = env.make_move(big_row, big_col, small_row, small_col, 1)
                    
                    next_state = env.get_state() if not done else None
                    agent.update_q_table(state, action, reward, next_state)
                    state = next_state
            else:
                # Tour de l'adversaire
                if opponent == 'random':
                    action = random.choice(valid_actions)
                else:
                    # On peut ajouter d'autres types d'adversaires ici
                    action = random.choice(valid_actions)
                
                if action is not None:
                    big_row, big_col, small_row, small_col = env.action_to_move(action)
                    reward, done, winner = env.make_move(big_row, big_col, small_row, small_col, -1)
                    state = env.get_state()
            
            current_player = -current_player
        
        # Compter les résultats
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
        
        # Afficher les progrès tous les 100 épisodes
        if (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100
            print(f"Épisode {episode + 1}/{num_episodes} - Victoires: {win_rate:.1f}% - Epsilon: {agent.epsilon:.3f}")
    
    print(f"\nEntraînement terminé:")
    print(f"Victoires: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")
    print(f"Défaites: {losses}/{num_episodes} ({losses/num_episodes*100:.1f}%)")
    print(f"Matchs nuls: {draws}/{num_episodes} ({draws/num_episodes*100:.1f}%)")
    
    return agent

# Fonction pour tester l'agent
def test_agent(agent, num_games=100):
    """Teste les performances de l'agent"""
    env = UltimateTicTacToeEnv()
    wins = 0
    draws = 0
    losses = 0
    
    print(f"Test de l'agent sur {num_games} parties...")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        current_player = 1
        
        while not done:
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            if current_player == 1:
                # L'agent joue sans exploration (epsilon = 0)
                action = agent.get_action(state, epsilon=0)
            else:
                # Adversaire aléatoire
                action = random.choice(valid_actions)
            
            if action is not None:
                big_row, big_col, small_row, small_col = env.action_to_move(action)
                reward, done, winner = env.make_move(big_row, big_col, small_row, small_col, current_player)
                state = env.get_state()
            
            current_player = -current_player
        
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"Résultats du test:")
    print(f"Victoires: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"Défaites: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"Matchs nuls: {draws}/{num_games} ({draws/num_games*100:.1f}%)")

# Exemple d'utilisation pour l'entraînement en ligne de commande
if __name__ == "__main__":
    print("=== Entraînement de l'IA Ultimate Tic-Tac-Toe ===")
    
    # Créer un agent
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.5)
    
    # Essayer de charger une Q-table existante
    try:
        agent.load_q_table("q_table.pkl")
        print("Q-table chargée avec succès!")
    except:
        print("Aucune Q-table trouvée, démarrage avec une table vide.")
    
    # Entraîner l'agent
    print("\nDémarrage de l'entraînement...")
    agent = train_agent(agent, num_episodes=5000)
    
    # Sauvegarder la Q-table
    agent.save_q_table("q_table.pkl")
    print("\nQ-table sauvegardée!")
    
    # Tester l'agent
    test_agent(agent, num_games=100)
    
    print("\nL'agent est prêt à jouer!")

# Classe utilitaire pour analyser les performances
class GameAnalyzer:
    def __init__(self):
        self.games_data = []
    
    def record_game(self, winner, num_moves, game_duration):
        """Enregistre les données d'une partie"""
        self.games_data.append({
            'winner': winner,
            'num_moves': num_moves,
            'duration': game_duration
        })
    
    def get_statistics(self):
        """Retourne des statistiques sur les parties enregistrées"""
        if not self.games_data:
            return {}
        
        wins_x = sum(1 for game in self.games_data if game['winner'] == 1)
        wins_o = sum(1 for game in self.games_data if game['winner'] == -1)
        draws = sum(1 for game in self.games_data if game['winner'] == 0)
        
        avg_moves = sum(game['num_moves'] for game in self.games_data) / len(self.games_data)
        avg_duration = sum(game['duration'] for game in self.games_data) / len(self.games_data)
        
        return {
            'total_games': len(self.games_data),
            'wins_x': wins_x,
            'wins_o': wins_o,
            'draws': draws,
            'win_rate_x': wins_x / len(self.games_data) * 100,
            'win_rate_o': wins_o / len(self.games_data) * 100,
            'draw_rate': draws / len(self.games_data) * 100,
            'avg_moves': avg_moves,
            'avg_duration': avg_duration
        }

# Fonction pour créer un agent pré-entraîné rapidement
def create_pretrained_agent():
    """Crée un agent avec quelques stratégies de base"""
    agent = QLearningAgent(learning_rate=0.05, discount_factor=0.9, epsilon=0.1)
    
    # On pourrait ici implémenter des heuristiques de base
    # pour donner à l'agent un bon point de départ
    
    return agent