import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
from ia_reinforcement import QLearningAgent, UltimateTicTacToeEnv

class UltimateTicTacToeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ultimate Tic-Tac-Toe")
        self.root.geometry("800x1000")  # Augmenté de 900 à 1000 pixels en hauteur
        
        # État du jeu
        self.env = UltimateTicTacToeEnv()
        self.agent = QLearningAgent()
        self.agent2 = QLearningAgent()  # Deuxième agent pour IA vs IA
        
        # Mode de jeu
        self.game_mode = tk.StringVar(value="pvp")  # pvp, pvc, aia, training
        self.current_player = 1
        self.game_active = True
        self.ai_vs_ai_speed = tk.IntVar(value=1000)  # Vitesse en ms entre les coups
        self.ai_vs_ai_running = False
        
        self.setup_ui()
        self.reset_game()
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre
        title = tk.Label(main_frame, text="Ultimate Tic-Tac-Toe", 
                        font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title.pack(pady=10)
        
        # Frame des contrôles
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(pady=10)
        
        # Sélection du mode
        tk.Label(control_frame, text="Mode de jeu:", font=('Arial', 12), 
                fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=5)
        
        modes = [("Joueur vs Joueur", "pvp"), 
                ("Joueur vs IA", "pvc"), 
                ("IA vs IA", "aia"),
                ("Entraîner l'IA", "training")]
        
        for text, mode in modes:
            tk.Radiobutton(control_frame, text=text, variable=self.game_mode, 
                          value=mode, font=('Arial', 10), fg='white', bg='#2c3e50',
                          selectcolor='#34495e', command=self.mode_changed).pack(side=tk.LEFT, padx=5)
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Nouvelle Partie", command=self.reset_game,
                 font=('Arial', 12), bg='#3498db', fg='white', padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Sauvegarder IA", command=self.save_agent,
                 font=('Arial', 12), bg='#27ae60', fg='white', padx=20).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Charger IA", command=self.load_agent,
                 font=('Arial', 12), bg='#f39c12', fg='white', padx=20).pack(side=tk.LEFT, padx=5)
        
        # Informations de jeu
        self.info_label = tk.Label(main_frame, text="", font=('Arial', 14),
                                  fg='white', bg='#2c3e50')
        self.info_label.pack(pady=10)
        
        # Frame pour l'entraînement
        self.training_frame = tk.Frame(main_frame, bg='#2c3e50')
        
        tk.Label(self.training_frame, text="Parties d'entraînement:", 
                font=('Arial', 12), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=5)
        
        self.training_games = tk.Entry(self.training_frame, width=10, font=('Arial', 12))
        self.training_games.insert(0, "1000")
        self.training_games.pack(side=tk.LEFT, padx=5)
        
        tk.Button(self.training_frame, text="Démarrer Entraînement", 
                 command=self.start_training, font=('Arial', 12), bg='#e74c3c', 
                 fg='white').pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.training_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10)
        
        # Frame pour IA vs IA
        self.ai_vs_ai_frame = tk.Frame(main_frame, bg='#2c3e50')
        
        tk.Label(self.ai_vs_ai_frame, text="Vitesse (ms):", 
                font=('Arial', 12), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=5)
        
        speed_scale = tk.Scale(self.ai_vs_ai_frame, from_=100, to=3000, 
                              orient=tk.HORIZONTAL, variable=self.ai_vs_ai_speed,
                              font=('Arial', 10), fg='white', bg='#2c3e50',
                              highlightbackground='#2c3e50', length=200)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.start_stop_ai_button = tk.Button(self.ai_vs_ai_frame, text="Démarrer Combat IA", 
                                             command=self.toggle_ai_vs_ai, font=('Arial', 12), 
                                             bg='#9b59b6', fg='white')
        self.start_stop_ai_button.pack(side=tk.LEFT, padx=10)
        
        tk.Button(self.ai_vs_ai_frame, text="Charger IA2", command=self.load_agent2,
                 font=('Arial', 12), bg='#e67e22', fg='white', padx=20).pack(side=tk.LEFT, padx=5)
        
        # Plateau de jeu
        self.game_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=3)
        self.game_frame.pack(pady=20)
        
        # Créer la grille 3x3 de sous-grilles
        self.buttons = {}
        self.board_frames = {}
        
        for big_row in range(3):
            for big_col in range(3):
                # Frame pour chaque sous-grille
                board_frame = tk.Frame(self.game_frame, bg='#2c3e50', 
                                     relief=tk.RAISED, bd=2)
                board_frame.grid(row=big_row, column=big_col, padx=2, pady=2)
                self.board_frames[(big_row, big_col)] = board_frame
                
                # Boutons dans chaque sous-grille
                for small_row in range(3):
                    for small_col in range(3):
                        btn = tk.Button(board_frame, text="", width=3, height=2,
                                      font=('Arial', 16, 'bold'), bg='#ecf0f1',
                                      command=lambda br=big_row, bc=big_col, 
                                               sr=small_row, sc=small_col: 
                                               self.make_move(br, bc, sr, sc))
                        btn.grid(row=small_row, column=small_col, padx=1, pady=1)
                        self.buttons[(big_row, big_col, small_row, small_col)] = btn
    
    def mode_changed(self):
        if self.game_mode.get() == "training":
            self.training_frame.pack(pady=10)
            self.ai_vs_ai_frame.pack_forget()
            self.game_frame.pack_forget()
        elif self.game_mode.get() == "aia":
            self.training_frame.pack_forget()
            self.ai_vs_ai_frame.pack(pady=10)
            self.game_frame.pack(pady=20)
        else:
            self.training_frame.pack_forget()
            self.ai_vs_ai_frame.pack_forget()
            self.game_frame.pack(pady=20)
        
        # Arrêter le combat IA si on change de mode
        if self.ai_vs_ai_running:
            self.toggle_ai_vs_ai()
            
        self.reset_game()
    
    def reset_game(self):
        self.env.reset()
        self.current_player = 1
        self.game_active = True
        
        # Réinitialiser l'affichage
        for key, btn in self.buttons.items():
            btn.config(text="", bg='#ecf0f1', state=tk.NORMAL)
        
        # Réinitialiser les couleurs des cadres
        for frame in self.board_frames.values():
            frame.config(bg='#2c3e50')
        
        self.update_info()
        self.highlight_active_boards()
        
        # Si mode IA vs IA et que le combat était en cours, le relancer
        if self.game_mode.get() == "aia" and self.ai_vs_ai_running:
            self.root.after(self.ai_vs_ai_speed.get(), self.ai_vs_ai_move)
    
    def make_move(self, big_row, big_col, small_row, small_col):
        if not self.game_active or self.game_mode.get() in ["training", "aia"]:
            return
        
        # Vérifier si le mouvement est valide
        if not self.env.is_valid_move(big_row, big_col, small_row, small_col):
            return
        
        # Faire le mouvement
        reward, done, winner = self.env.make_move(big_row, big_col, small_row, small_col, self.current_player)
        
        # Mettre à jour l'affichage
        self.update_display()
        
        if done:
            self.game_over(winner)
            return
        
        # Changer de joueur
        self.current_player = -self.current_player
        
        # Si c'est le mode contre l'IA et c'est au tour de l'IA
        if self.game_mode.get() == "pvc" and self.current_player == -1:
            self.root.after(500, self.ai_move)  # Délai pour l'IA
        
        self.update_info()
        self.highlight_active_boards()
    
    def ai_move(self):
        if not self.game_active:
            return
        
        # L'IA fait son mouvement
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            print("Aucune action valide pour l'IA")
            return
        
        action = self.agent.get_action(state, epsilon=0.1)  # Peu d'exploration en jeu
        
        if action is None:
            print("L'IA n'a pas pu choisir d'action")
            # En dernier recours, choisir une action aléatoire
            action = random.choice(valid_actions)
        
        big_row, big_col, small_row, small_col = self.env.action_to_move(action)
        reward, done, winner = self.env.make_move(big_row, big_col, small_row, small_col, -1)
        
        self.update_display()
        
        if done:
            self.game_over(winner)
            return
        
        self.current_player = 1  # Retour au joueur humain
        
        self.update_info()
        self.highlight_active_boards()
    
    def toggle_ai_vs_ai(self):
        """Démarre ou arrête le combat IA vs IA"""
        if not self.ai_vs_ai_running:
            self.ai_vs_ai_running = True
            self.start_stop_ai_button.config(text="Arrêter Combat IA", bg='#e74c3c')
            self.reset_game()
            self.root.after(self.ai_vs_ai_speed.get(), self.ai_vs_ai_move)
        else:
            self.ai_vs_ai_running = False
            self.start_stop_ai_button.config(text="Démarrer Combat IA", bg='#9b59b6')
    
    def ai_vs_ai_move(self):
        """Gère un coup dans le mode IA vs IA"""
        if not self.ai_vs_ai_running or not self.game_active:
            return
        
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            print("Aucune action valide")
            return
        
        # Choisir l'agent selon le joueur actuel
        if self.current_player == 1:
            agent = self.agent
            agent_name = "IA1 (X)"
        else:
            agent = self.agent2
            agent_name = "IA2 (O)"
        
        # L'IA choisit son action (avec peu d'exploration pour un jeu plus déterministe)
        action = agent.get_action(state, epsilon=0.05)
        
        if action is None:
            print(f"{agent_name} n'a pas pu choisir d'action")
            action = random.choice(valid_actions)
        
        # Effectuer le mouvement
        big_row, big_col, small_row, small_col = self.env.action_to_move(action)
        reward, done, winner = self.env.make_move(big_row, big_col, small_row, small_col, self.current_player)
        
        # Mettre à jour l'affichage
        self.update_display()
        
        if done:
            self.game_over(winner)
            # Relancer automatiquement une nouvelle partie après 2 secondes
            if self.ai_vs_ai_running:
                self.root.after(2000, self.reset_game)
            return
        
        # Changer de joueur
        self.current_player = -self.current_player
        
        self.update_info()
        self.highlight_active_boards()
        
        # Programmer le prochain mouvement
        if self.ai_vs_ai_running:
            self.root.after(self.ai_vs_ai_speed.get(), self.ai_vs_ai_move)
    
    def update_display(self):
        # Mettre à jour tous les boutons
        for big_row in range(3):
            for big_col in range(3):
                for small_row in range(3):
                    for small_col in range(3):
                        value = self.env.small_boards[big_row][big_col][small_row][small_col]
                        btn = self.buttons[(big_row, big_col, small_row, small_col)]
                        
                        if value == 1:
                            btn.config(text="X", fg='#e74c3c', bg='#ecf0f1')
                        elif value == -1:
                            btn.config(text="O", fg='#3498db', bg='#ecf0f1')
                        else:
                            btn.config(text="", bg='#ecf0f1')
                
                # Vérifier si la sous-grille est gagnée
                winner = self.env.check_small_board_winner(big_row, big_col)
                frame = self.board_frames[(big_row, big_col)]
                if winner == 1:
                    frame.config(bg='#e74c3c')  # Rouge pour X
                elif winner == -1:
                    frame.config(bg='#3498db')  # Bleu pour O
                elif self.env.is_small_board_full(big_row, big_col):
                    frame.config(bg='#95a5a6')  # Gris pour égalité
    
    def highlight_active_boards(self):
        # Réinitialiser toutes les couleurs
        for frame in self.board_frames.values():
            if frame.cget('bg') == '#2c3e50':  # Seulement si pas encore gagnée
                frame.config(bg='#2c3e50')
        
        # Surligner les plateaux actifs
        for big_row, big_col in self.env.get_valid_boards():
            frame = self.board_frames[(big_row, big_col)]
            if frame.cget('bg') == '#2c3e50':
                frame.config(bg='#f39c12')  # Orange pour actif
    
    def update_info(self):
        if not self.game_active:
            return
        
        mode_text = {"pvp": "Joueur vs Joueur", "pvc": "Joueur vs IA", 
                    "aia": "IA vs IA", "training": "Entraînement"}
        current_mode = mode_text[self.game_mode.get()]
        
        if self.game_mode.get() == "pvc":
            player_text = "Votre tour (X)" if self.current_player == 1 else "Tour de l'IA (O)"
        elif self.game_mode.get() == "aia":
            player_text = "Tour de l'IA1 (X)" if self.current_player == 1 else "Tour de l'IA2 (O)"
        else:
            player_text = f"Tour du joueur {'X' if self.current_player == 1 else 'O'}"
        
        self.info_label.config(text=f"{current_mode} | {player_text}")
    
    def game_over(self, winner):
        self.game_active = False
        
        if self.game_mode.get() == "aia":
            if winner == 1:
                message = "Victoire de l'IA1 (X) !"
            elif winner == -1:
                message = "Victoire de l'IA2 (O) !"
            else:
                message = "Match nul entre les IA !"
        elif self.game_mode.get() == "pvc":
            if winner == 1:
                message = "Vous avez gagné !"
            elif winner == -1:
                message = "L'IA a gagné !"
            else:
                message = "Match nul !"
        else:  # pvp
            if winner == 1:
                message = "Victoire de X !"
            elif winner == -1:
                message = "Victoire de O !"
            else:
                message = "Match nul !"
        
        self.info_label.config(text=message)
        
        # Ne pas afficher de popup en mode IA vs IA continu
        if self.game_mode.get() != "aia" or not self.ai_vs_ai_running:
            messagebox.showinfo("Fin de partie", message)
    
    def start_training(self):
        try:
            num_games = int(self.training_games.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide de parties")
            return
        
        self.progress.config(maximum=num_games, value=0)
        self.root.update()
        
        # Entraînement
        for i in range(num_games):
            self.env.reset()
            done = False
            current_player = 1
            
            while not done:
                state = self.env.get_state()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions:
                    break
                
                if current_player == 1:
                    # L'agent apprend en tant que joueur 1
                    action = self.agent.get_action(state, epsilon=0.3)
                else:
                    # Joueur aléatoire ou autre agent
                    action = np.random.choice(valid_actions)
                
                if action is not None:
                    big_row, big_col, small_row, small_col = self.env.action_to_move(action)
                    reward, done, winner = self.env.make_move(big_row, big_col, small_row, small_col, current_player)
                    
                    if current_player == 1:
                        next_state = self.env.get_state() if not done else None
                        self.agent.update_q_table(state, action, reward, next_state)
                
                current_player = -current_player
            
            # Mettre à jour la barre de progression
            if i % 50 == 0:
                self.progress.config(value=i)
                self.root.update()
        
        self.progress.config(value=num_games)
        messagebox.showinfo("Entraînement terminé", f"L'IA a été entraînée sur {num_games} parties")
    
    def save_agent(self):
        try:
            self.agent.save_q_table("q_table.npy")
            messagebox.showinfo("Sauvegarde", "L'IA a été sauvegardée avec succès")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def load_agent(self):
        try:
            self.agent.load_q_table("q_table.npy")
            messagebox.showinfo("Chargement", "L'IA a été chargée avec succès")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}")
    
    def load_agent2(self):
        """Charge une IA différente pour l'agent 2"""
        try:
            self.agent2.load_q_table("q_table2.npy")
            messagebox.showinfo("Chargement", "L'IA2 a été chargée avec succès")
        except Exception as e:
            try:
                # Si pas de q_table2, essayer de charger la même que l'agent 1
                self.agent2.load_q_table("q_table.npy")
                messagebox.showinfo("Chargement", "L'IA2 chargée avec la même table que l'IA1")
            except Exception as e2:
                messagebox.showerror("Erreur", f"Erreur lors du chargement de l'IA2: {str(e2)}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    game = UltimateTicTacToeGUI()
    game.run()