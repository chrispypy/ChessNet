import chess
import cython_chess
import numpy as np
import tensorflow as tf


C_PUCT = 1.5           # Exploration-Konstante für PUCT
VIRTUAL_LOSS = 3.0      # Virtual Loss Wert für Batched MCTS
BATCH_SIZE = 16          # Anzahl Leafs pro Batch


class MCTSNode:
    def __init__(self, board: chess.Board | None = None, parent=None, move=None, parent_action_idx=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.parent_action_idx = parent_action_idx
        
        self.children: dict[chess.Move, 'MCTSNode'] = {}
        self.legal_moves = []
        self.child_priors = np.array([], dtype=np.float32)
        self.child_visits = np.array([], dtype=np.float32)
        
        self.visit_count = 0.0
        self.value_sum = 0.0
        self.prior = 0.0
        
        self.is_terminal_win = False
        self.fatal_moves = set()
        
        # Board nur lazy erstellen, wenn leaf selected ist
        if self.board is not None:
            self._init_board_state()
        else:
            self.is_terminal = False
            self.terminal_value = None

    def is_repetition(self):
        """Prüft auf Wiederholungen im MCTS-Baum UND in der echten Zughistorie."""
        if self.board is None:
            return False
        curr_key = self.board._transposition_key()
        node = self.parent
        while node is not None:
            if node.board is not None and node.board._transposition_key() == curr_key:
                return True
            if hasattr(node, "history_set") and curr_key in node.history_set:
                return True
            node = node.parent
        return False

    def _init_board_state(self):
        self.legal_moves = list(cython_chess.generate_legal_moves(self.board, chess.BB_ALL, chess.BB_ALL))
        
        is_rep = self.is_repetition()
                
        self.is_terminal = (len(self.legal_moves) == 0) or (self.board.halfmove_clock >= 100) or is_rep or self.board.is_insufficient_material()
        
        if self.is_terminal:
            self.terminal_value = self._get_terminal_value()
            if self.terminal_value == -1.0 and self.parent is not None:
                self.parent.is_terminal_win = True
                if self.parent.parent is not None:
                    self.parent.parent.fatal_moves.add(self.parent.move)
        else:
            self.terminal_value = None

    def ensure_board(self):
        if self.board is None:
            self.board = self.parent.board.copy()
            self.board.push(self.move)
            self._init_board_state()

    def update_stats(self, visit_delta, value_delta):
        self.visit_count += visit_delta
        self.value_sum += value_delta
        if self.parent is not None and self.parent_action_idx is not None:
            self.parent.child_visits[self.parent_action_idx] += visit_delta

    def is_expanded(self):
        return len(self.children) > 0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return -self.parent.q_value if self.parent else 0.0
        return self.value_sum / self.visit_count

    def _get_terminal_value(self):
        if len(self.legal_moves) == 0 and self.board.is_check():
            return -1.0  # Matt
        return 0.0  # Patt oder Remis

    def best_child_puct_idx(self):
        if self.is_terminal_win:
            for i, move in enumerate(self.legal_moves):
                if move in self.children:
                    child = self.children[move]
                    if child.is_terminal and child.terminal_value == -1.0:
                        return i

        q_values = np.zeros(len(self.legal_moves), dtype=np.float32)
        for i, move in enumerate(self.legal_moves):
            if move in self.children:
                q_values[i] = -self.children[move].q_value
                if move in self.fatal_moves:
                    q_values[i] = -np.inf
        
        U = C_PUCT * self.child_priors * np.sqrt(max(1.0, self.visit_count)) / (1.0 + self.child_visits)
        puct = q_values + U
        return np.argmax(puct)

class MCTS:
    def __init__(self, model, encoder, board_encoder):
        self.model = model
        self.encoder = encoder
        self.board_encoder = board_encoder
        
        if not hasattr(self.model, '_fast_predict'):
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, 8, 8, 6), dtype=tf.float32)])
            def fast_predict(x):
                return self.model(x, training=False)
            self.model._fast_predict = fast_predict
            
        self._fast_predict = self.model._fast_predict

    # ─────────────────────────────────────────────
    # Hauptmethode: Führe N Simulationen durch
    # ─────────────────────────────────────────────
    def search(self, root_board: chess.Board, n_simulations: int, early_stopping: bool = False, add_noise: bool = True) -> np.ndarray:
        """
        Führe MCTS-Suche durch und gib die Visit-Count-Verteilung zurück.

        Returns:
            visits: np.ndarray mit Shape (n_legal_moves,) – normalisierte Besuchszahlen
            moves:  Liste der legalen Züge in der gleichen Reihenfolge
            root_eval: float – P(win) - P(loss) aus Sicht des Side-to-move am Root
        """
        history_set = set()
        tmp_board = root_board.copy()
        while tmp_board.move_stack:
            tmp_board.pop()
            history_set.add(tmp_board._transposition_key())

        root = MCTSNode(root_board)
        root.history_set = history_set  
        
        root_eval = self._expand_node_single(root)

        # wenn nur ein legaler Zug -> fertig
        if len(root.legal_moves) == 1:
            visits = np.array([1.0], dtype=np.float32)
            return visits, root.legal_moves, root_eval

        # Exploration: Dirichlet-Rauschen auf Root-Node Priors anwenden
        if add_noise:
            dirichlet_alpha = 0.3
            dirichlet_epsilon = 0.25
            if root.legal_moves:
                noise = np.random.dirichlet([dirichlet_alpha] * len(root.legal_moves))
                root.child_priors = (1 - dirichlet_epsilon) * root.child_priors + dirichlet_epsilon * noise
                for i, move in enumerate(root.legal_moves):
                    root.children[move].prior = root.child_priors[i]

        sims_done = 0
        while sims_done < n_simulations:
            
            # ── Early Stopping (Beschleunigung in gierigen Phasen) ──
            if early_stopping and sims_done > 0 and len(root.child_visits) > 1:
                sorted_v = np.sort(root.child_visits)
                if sorted_v[-1] > sorted_v[-2] + (n_simulations - sims_done):
                    break

            # ── Phase 1: Sammle Leafs für einen Batch ──
            leaves = []
            paths = []

            for _ in range(min(BATCH_SIZE, n_simulations - sims_done)):
                leaf, path = self._select_leaf(root)

                # Fall 1: Terminal-Knoten -> sofort backprop
                if leaf.is_terminal:
                    self._backpropagate(path, leaf.terminal_value)
                    sims_done += 1
                    continue

                # Fall 2: Kollision (Knoten schon expanded von anderem Pfad in diesem Batch)
                if leaf.is_expanded():
                    self._remove_virtual_loss(path)
                    continue

                leaves.append(leaf)
                paths.append(path)

            if not leaves:
                continue

            batch_in = np.stack([self.board_encoder.encode(leaf.board) for leaf in leaves], axis=0)
            result = self._fast_predict(tf.constant(batch_in))
            policy_batch = result[0].numpy()
            value_batch = result[1].numpy()

            # ── Phase 3: Expand & Backprop ──
            for i, (leaf, path) in enumerate(zip(leaves, paths)):
                policy_logits = policy_batch[i]
                value_probs = value_batch[i]

                # Expandiere den Leaf-Knoten mit der Policy vom Netz
                self._expand_node(leaf, policy_logits)

                v = value_probs[0] - value_probs[2]  # P(win) - P(loss)

                self._backpropagate(path, v)
                sims_done += 1

        self._last_root = root  # Für UCI: Per-Zug Q-Values
        visits = root.child_visits.copy()
        if visits.sum() > 0:
            return visits / visits.sum(), root.legal_moves, root_eval
        return np.ones_like(visits) / len(visits), root.legal_moves, root_eval

    # ─────────────────────────────────────────────
    # Selektion: Laufe den Baum runter bis zu einem Leaf
    # ─────────────────────────────────────────────
    def _select_leaf(self, node: MCTSNode):
        path = [node]
        while node.is_expanded() and not node.is_terminal:
            best_idx = node.best_child_puct_idx()
            best_move = node.legal_moves[best_idx]
            child = node.children[best_move]
            
            # BOARD LAZY LADEN 
            child.ensure_board()
            path.append(child)
            child.update_stats(1, VIRTUAL_LOSS)
            node = child
        return node, path
        
    # ─────────────────────────────────────────────
    # Expansion: Erzeuge Kindknoten aus Netz-Policy
    # ─────────────────────────────────────────────
    def _expand_node(self, node: MCTSNode, policy_logits: np.ndarray):
        """
        Expandiere einen Knoten: Erzeuge für jeden legalen Zug einen Kindknoten
        und weise ihm den Prior aus der Netz-Policy zu.
        """
        # Softmax nur über legale Züge (statt über alle 3584)
        legal_indices = self.encoder.encode_batch(node.legal_moves)
        legal_logits = policy_logits[legal_indices]
        legal_logits = legal_logits - legal_logits.max()
        priors = np.exp(legal_logits, dtype=np.float32)
        priors /= priors.sum()

        num_children = len(node.legal_moves)
        node.child_priors = priors
        node.child_visits = np.zeros(num_children, dtype=np.float32)

        for i, move in enumerate(node.legal_moves):
            child = MCTSNode(board=None, parent=node, move=move, parent_action_idx=i)
            child.prior = priors[i]
            node.children[move] = child

    def _expand_node_single(self, node: MCTSNode):
        """Convenience: Expandiere Root-Knoten mit einzelner Netz-Evaluation. Gibt root_eval zurück."""
        board_tensor = self.board_encoder.encode(node.board)
        batch_in = np.expand_dims(board_tensor, axis=0)
        result = self._fast_predict(tf.constant(batch_in))
        self._expand_node(node, result[0].numpy()[0])
        value_probs = result[1].numpy()[0]
        return float(value_probs[0] - value_probs[2])  # P(win) - P(loss)

    # ─────────────────────────────────────────────
    # Backpropagation: Wert den Baum hochschicken
    # ─────────────────────────────────────────────
    def _backpropagate(self, path: list[MCTSNode], value: float):
        for node in reversed(path[1:]):
            node.update_stats(0, value - VIRTUAL_LOSS)
            value = -value
        path[0].update_stats(1, value)

    def _remove_virtual_loss(self, path: list[MCTSNode]):
        """Entferne Virtual Loss bei Kollision (kein echtes Backprop)."""
        for node in path[1:]:
            node.update_stats(-1, -VIRTUAL_LOSS)

    def pick_move(self, root_board: chess.Board, n_simulations: int, temperature: float = 1.0):
        """
        Führe MCTS-Suche durch und wähle einen Zug.

        temperature > 0: Zug wird proportional zu visit_count^(1/temp) gesampelt
        temperature = 0: Greedy, wähle den meistbesuchten Zug
        """
        visits, moves, _ = self.search(root_board, n_simulations, early_stopping=(temperature == 0))

        if temperature == 0:
            return moves[np.argmax(visits)]

        else:
            adjusted = visits ** (1.0/temperature)
            probs = adjusted / adjusted.sum()
            return moves[np.random.choice(len(moves), p=probs)]
