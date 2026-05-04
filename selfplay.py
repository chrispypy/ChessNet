import chess
import chess.syzygy
import numpy as np
import keras
import os
import json
import csv
from collections import deque
from encoder import Encoder, BoardEncoder
from mcts import MCTS
from game_logger import GameLogger
from chessnet import build_chessnet
import tensorflow as tf

# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────
N_GAMES = 50             # Spiele pro Trainingsiteration
N_SIMULATIONS = 800      # MCTS-Simulationen pro Zug
N_ITERATIONS = 5000      # Große Trainingsschleifen
TEMP_THRESHOLD = 0      # Erste 15 Halbzüge mit Temperatur
BATCH_SIZE = 512         # Trainings-Batchsize
MAX_MOVES = 200          # Maximale Züge pro Spiel (Remis bei Überschreitung -> Reduziert sinnloses Herumgeschiebe)
BUFFER_SIZE = 100_000     # Max Positionen im Replay Buffer
TRAIN_SAMPLES = 8192     # Positionen pro Trainings-Epoche (hoch wegen 4x Augmentation)
N_BATTLE = 20            # Spiele im Battle-Modus
SYZYGY_PATH = os.path.join(os.path.dirname(__file__), 'syzygy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RES_DICT = {'1-0':     1,
            '0-1':    -1,
            '1/2-1/2': 0}

# Curriculum: (max_dtz, win_rate_zum_aufsteigen, endgame_type)
# Wenn die Win-Rate über den Threshold geht, nächste Phase 
CURRICULUM = [
    (8,   0.95, {'KQvK': 1.0}),                                  # Schneller Bootstrap: Lerne was Matt ist
    (999, 0.90, {'KQvK': 1.0}),                                  # Dame aus zufälligen Stellungen
    (999, 0.90, {'KRvK': 0.8, 'KQvK': 0.2}),                     # Turm-Matt
    (999, 0.90, {'KQvKR': 0.85, 'KRvK': 0.15, 'KQvK': 0.0}),     # Volles KQvKR Endspiel
]

# --- Altes DTZ-Curriculum (auskommentiert wegen Edge-Bias) ---
# CURRICULUM = [
#     # --- STUFE 1: Dame ---
#     (1,   0.95, {'KQvK': 1.0}),   
#     (3,   0.95, {'KQvK': 1.0}),   
#     (8,   0.90, {'KQvK': 1.0}),   
#     (999, 0.90, {'KQvK': 1.0}),   
#
#     # --- STUFE 2: Turm (Dame wird zu 20% beigemischt, um sie nicht zu vergessen) ---
#     (1,   0.95, {'KRvK': 0.8, 'KQvK': 0.2}),   
#     (3,   0.95, {'KRvK': 0.8, 'KQvK': 0.2}),   
#     (8,   0.90, {'KRvK': 0.8, 'KQvK': 0.2}),   
#     (25,  0.90, {'KRvK': 0.8, 'KQvK': 0.2}),   
#     (999, 0.90, {'KRvK': 0.8, 'KQvK': 0.2}),   
#
#     # --- STUFE 3:---
#     (1,   0.7, {'KQvKR': 0.7, 'KRvK': 0.15, 'KQvK': 0.15}), 
#     (3,   0.7, {'KQvKR': 0.7, 'KRvK': 0.15, 'KQvK': 0.15}), 
#     (8,   0.85, {'KQvKR': 0.7, 'KRvK': 0.15, 'KQvK': 0.15}), 
#     (15,  0.85, {'KQvKR': 0.8, 'KRvK': 0.1,  'KQvK': 0.1}),  
#     (30,  0.85, {'KQvKR': 0.9, 'KRvK': 0.05, 'KQvK': 0.05}), 
#     (999, 0.95, {'KRvK': 0.8, 'KQvK': 0.2}),
#     (999, 0.90, {'KQvKR': 0.6, 'KRvK': 0.2, 'KQvK': 0.2}),
# ]


# ─────────────────────────────────────────────
# Zufällige Startposition generieren
# ─────────────────────────────────────────────
def generate_random_endgame(tablebase=None, max_dtz=999, queen_color=None, endgame_dict=None):
    """
    Generiere eine zufällige legale Position basierend auf dem Curriculum.
    Gibt (board, queen_color, endgame_type, expected_winner) zurück.
    expected_winner: chess.WHITE / chess.BLACK / None (Remis-Stellung)
    """
    king_w = chess.Piece(chess.KING, chess.WHITE)
    king_b = chess.Piece(chess.KING, chess.BLACK)
    
    # Nur überschreiben, wenn KEINE feste Farbe vom Battle-Loop vorliegt
    if queen_color is None:
        queen_color = bool(np.random.choice([True, False]))
        
    pieces = [king_w, king_b]
    if endgame_dict is None:
        endgame_dict = {'KQvK': 1.0}
    types = list(endgame_dict.keys())
    probs = list(endgame_dict.values())
    endgame_type = np.random.choice(types, p=probs)
    
    if endgame_type == 'KQvK':
        pieces.append(chess.Piece(chess.QUEEN, queen_color))
    elif endgame_type == 'KRvK':
        pieces.append(chess.Piece(chess.ROOK, queen_color))
    elif endgame_type in ('KQvKR', 'KQvKR_rook'):
        pieces.append(chess.Piece(chess.QUEEN, queen_color))
        pieces.append(chess.Piece(chess.ROOK, not queen_color))

    while True:
        board = chess.Board(None)
        board.clear()
        squares = np.random.choice(range(64), len(pieces), replace=False)
        for i, square in enumerate(squares):
            board.set_piece_at(int(square), pieces[i])
        board.turn = bool(np.random.choice([True, False]))
        board.set_castling_fen('-')

        if not board.is_valid():
            continue
        if board.is_game_over():
            continue

        # DTZ filtern wenn Tablebase verfügbar
        expected_winner = None
        if tablebase is not None:
            try:
                dtz = tablebase.probe_dtz(board)
                # KQvKR_rook: Stellungen wo die Turmseite hält oder gewinnt (DTZ <= 0)
                if endgame_type == 'KQvKR_rook':
                    if dtz > 0:  # Dame-Seite gewinnt → überspringen
                        continue
                elif max_dtz < 999 and (abs(dtz) > max_dtz or dtz == 0):
                    continue
                # Erwarteten Gewinner bestimmen
                if dtz > 0:
                    expected_winner = board.turn
                elif dtz < 0:
                    expected_winner = not board.turn
            except chess.syzygy.MissingTableError:
                if endgame_type == 'KQvKR_rook':
                    continue
                pass

        return board, queen_color, endgame_type, expected_winner


# ─────────────────────────────────────────────
# Ein einzelnes Spiel spielen
# ─────────────────────────────────────────────
def play_game(mcts: MCTS, logger: GameLogger, tablebase=None, max_dtz=999, endgame_dict=None):
    """
    Spiele ein komplettes Spiel mit MCTS und sammle Trainingsdaten.
    Gibt (game_data, result, queen_color, endgame_type, expected_winner) zurück.
    """
    board, queen_color, endgame_type, expected_winner = generate_random_endgame(tablebase, max_dtz, endgame_dict=endgame_dict)
    logger.new_game(board)
    game_data = []
    move_count = 0
    board_encoder = mcts.board_encoder
    move_encoder = mcts.encoder

    while not board.is_game_over(claim_draw=True) and move_count < MAX_MOVES:
        temperature = 1 if move_count < TEMP_THRESHOLD else 0
        
        # Dynamische Reduzierung der Simulationen für triviale Endspiele (KQvK, KRvK) (erstmal deaktiviert, wegen schlechten turm endspielen)
        if len(board.piece_map()) <= 3:
            current_sims = min(N_SIMULATIONS, 800)
        else:
            current_sims = N_SIMULATIONS

        visits, moves, root_eval = mcts.search(board, current_sims)

        board_encoding = board_encoder.encode(board)
        policy_target = np.zeros(move_encoder.policy_size)
        for visit, move in zip(visits, moves):
            policy_target[move_encoder.encode(move)] = visit
        turn = board.turn
        game_data.append((board_encoding, policy_target, turn))

        # Eval aus MCTS-Root (kein extra Netz-Aufruf nötig!)
        eval_white = root_eval if turn else -root_eval

        if temperature == 0:
            chosen_move = moves[np.argmax(visits)]
        else:
            adjusted = visits ** (1.0 / temperature)
            probs = adjusted / adjusted.sum()
            chosen_move = moves[np.random.choice(len(moves), p=probs)]
        board.push(chosen_move)
        move_count += 1
        logger.log_move(board, chosen_move, eval_score=eval_white)

    if move_count >= MAX_MOVES:
        result = '1/2-1/2'
    else:
        result = board.result(claim_draw=True)
    logger.game_over(board, result)
    return game_data, result, queen_color, endgame_type, expected_winner


# ─────────────────────────────────────────────
# Battle: Spiel zwischen zwei Modellen
# ─────────────────────────────────────────────
def battle_game(white_mcts: MCTS, black_mcts: MCTS, start_board: chess.Board):
    """Ein Spiel zwischen zwei MCTS-Engines. Gibt Ergebnis als String zurück."""
    board = start_board.copy()
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < MAX_MOVES:
        # Wähle Engine je nach Farbe
        mcts = white_mcts if board.turn else black_mcts
        visits, moves, _ = mcts.search(board, N_SIMULATIONS, early_stopping=True)
        chosen_move = moves[np.argmax(visits)]  # Greedy im Battle
        board.push(chosen_move)
        move_count += 1

    if move_count >= MAX_MOVES:
        return '1/2-1/2'
    return board.result(claim_draw=True)


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self._encoder = Encoder()  # Für Augmentation-LUTs

    def add_game(self, game_data, result):
        """Füge alle Positionen eines Spiels zum Buffer hinzu (inkl. Augmentation)."""
        res = RES_DICT[result]
        white_win = float(res == 1)
        black_win = float(res == -1)

        discount = 0.99  # DISCOUNT: Jeder Zug weiter weg vom Matt kostet 1% Win-Probability

        current_w = white_win
        current_b = black_win

        # Rückwärts iterieren, um nahe am Matt die vollen 100% zu vergeben
        for board_enc, policy_target, turn in reversed(game_data):
            # Der Decay blutet in die Draw-Probability (Sicherheit nimmt ab)
            current_draw = 1.0 - current_w - current_b

            value_target = np.array([
                turn * current_w + (not turn) * current_b,
                current_draw,
                turn * current_b + (not turn) * current_w
            ], dtype=np.float32)
            
            # Original
            self.buffer.append((board_enc, policy_target, value_target))
            
            # 3x Augmentation (H-Flip, V-Flip, 180° Rotation)
            aug_boards = BoardEncoder.augment_board(board_enc)
            aug_policies = self._encoder.augment_policy(policy_target)
            for ab, ap in zip(aug_boards, aug_policies):
                self.buffer.append((ab, ap, value_target))

            current_w *= discount
            current_b *= discount

    def sample(self, n=TRAIN_SAMPLES):
        """Zufälliges Sample aus dem Buffer."""
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in indices]
        boards, policies, values = zip(*batch)
        return np.array(boards), np.array(policies), np.array(values)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────
def save_checkpoint(model, iteration, phase, recent_results, is_best=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(os.path.join(CHECKPOINT_DIR, 'model_latest.keras'))
    if is_best:
        model.save(os.path.join(CHECKPOINT_DIR, 'model_best.keras'))
        
    state = {
        'iteration': iteration,
        'phase': phase,
        'recent_results': list(recent_results),
    }
    with open(os.path.join(CHECKPOINT_DIR, 'state.json'), 'w') as f:
        json.dump(state, f)
    print(f"  Checkpoint gespeichert (Iter {iteration+1}, Phase {phase+1})")

def log_metrics(metrics: dict):
    filepath = os.path.join(CHECKPOINT_DIR, 'training_metrics.csv')
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def load_checkpoint():
    state_path = os.path.join(CHECKPOINT_DIR, 'state.json')
    model_path = os.path.join(CHECKPOINT_DIR, 'model_latest.keras')
    if not os.path.exists(state_path):
        return None
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, 'model_best.keras')
        if not os.path.exists(model_path):
            return None
            
    model = keras.models.load_model(model_path)
    
    # Modell mit aktueller Loss-Konfiguration neu kompilieren,
    # damit Änderungen an chessnet.py (z.B. Policy-Weight) wirksam werden
    from keras.optimizers import Adam
    from keras.losses import CategoricalCrossentropy
    from chessnet import LR, REG_CONST
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss={
            'policy_output': CategoricalCrossentropy(from_logits=True),
            'value_output': CategoricalCrossentropy(from_logits=False),
        },
        loss_weights={
            'policy_output': 2.0,
            'value_output': 1.0,
        }
    )
    
    with open(state_path) as f:
        state = json.load(f)
        
    best_path = os.path.join(CHECKPOINT_DIR, 'model_best.keras')
    if os.path.exists(best_path):
        best_model = keras.models.load_model(best_path)
        best_weights = best_model.get_weights()
    else:
        best_weights = model.get_weights()
        
    return model, state, best_weights


# ─────────────────────────────────────────────
# Hauptschleife: Self-Play + Training
# ─────────────────────────────────────────────
def train():
    # Checkpoint laden oder frisch starten
    checkpoint = load_checkpoint()
    if checkpoint:
        model, state, best_weights = checkpoint
        start_iter = state['iteration'] + 1
        phase = state['phase']
        if phase >= len(CURRICULUM): phase = len(CURRICULUM) - 1 # Fallback für alte Saves
        recent_results = deque(state['recent_results'], maxlen=50)
        max_dtz, win_threshold, endgame_dict = CURRICULUM[phase]
        # LR passend zur aktuellen Phase setzen
        if phase >= 3:
            model.optimizer.learning_rate.assign(1e-4)
        elif phase >= 2:
            model.optimizer.learning_rate.assign(5e-4)
        print(f"Checkpoint geladen: Iteration {start_iter}, Phase {phase+1} ({endgame_dict}, DTZ={max_dtz}, LR={float(model.optimizer.learning_rate):.0e})")
    else:
        model = build_chessnet()
        best_weights = model.get_weights()
        start_iter = 0
        phase = 0
        max_dtz, win_threshold, endgame_dict = CURRICULUM[phase]
        recent_results = deque(maxlen=50)
        print("Neues Training gestartet.")

    encoder = Encoder()
    board_encoder = BoardEncoder()
    mcts = MCTS(model, encoder, board_encoder)
    logger = GameLogger()
    replay_buffer = ReplayBuffer()

    model_old = build_chessnet()
    model_old.set_weights(best_weights)
    mcts_old = MCTS(model_old, encoder, board_encoder)

    # Syzygy Tablebase laden
    try:
        tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
        print(f"Syzygy Tablebase geladen aus {SYZYGY_PATH}")
    except Exception as e:
        print(f"Keine Tablebase gefunden: {e}")
        tablebase = None
    
    for iteration in range(start_iter, N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{N_ITERATIONS}")
        print(f"  Curriculum Phase {phase + 1}/{len(CURRICULUM)} (max DTZ={max_dtz})")
        print(f"{'='*50}")
        epochs = min(30, max(5, len(replay_buffer) // 2000))

        # ── Setup Epoch Metrics ──
        epoch_metrics = {
            'iteration': iteration + 1,
            'phase': phase + 1,
            'endgame_type': str(endgame_dict),
            'max_dtz': max_dtz,
            'avg_game_length': 0.0,
            'win_rate': 0.0,
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'battle_new_wins': 0,
            'battle_old_wins': 0,
            'battle_draws': 0,
            'upgraded_phase': False
        }

        # ── Phase 1: Self-Play ──
        game_lengths = []
        primary_type = max(endgame_dict, key=endgame_dict.get)  # Haupttyp für Win-Rate
        for game_num in range(N_GAMES):
            game_data, result, queen_color, endgame_type, expected_winner = play_game(mcts, logger, tablebase, max_dtz, endgame_dict)
            replay_buffer.add_game(game_data, result)
            game_lengths.append(len(game_data))

            # Curriculum: Win-Rate anhand Tablebase-Erwartung tracken (nur Haupttyp)
            if endgame_type == primary_type and expected_winner is not None:
                actual_winner = None
                if result == '1-0':
                    actual_winner = chess.WHITE
                elif result == '0-1':
                    actual_winner = chess.BLACK
                recent_results.append(actual_winner == expected_winner)

            if (game_num + 1) % 10 == 0:
                print(f"  Spiel {game_num + 1}/{N_GAMES} abgeschlossen")

        epoch_metrics['avg_game_length'] = sum(game_lengths) / max(1, len(game_lengths))
        print(f"  Buffer: {len(replay_buffer)} Positionen")
        if len(replay_buffer) < BATCH_SIZE * 4:
            print('Zu wenig Positionen im Buffer, überspringe Training')
            continue

        # ── Phase 2: Training auf Buffer ──
        final_value_loss = 0.0
        final_policy_loss = 0.0
        final_loss = 0.0
        for epoch in range(epochs):
            boards, policies, values = replay_buffer.sample(TRAIN_SAMPLES)
            history = model.fit(
                boards,
                {'policy_output': policies, 'value_output': values},
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=0
            )
            final_value_loss = history.history['value_output_loss'][0]
            final_policy_loss = history.history['policy_output_loss'][0]
            final_loss = history.history['loss'][0]
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} – V-Loss: {final_value_loss:.3f}, P-Loss: {final_policy_loss:.3f}")
        epoch_metrics['loss'] = final_loss
        epoch_metrics['policy_loss'] = final_policy_loss
        epoch_metrics['value_loss'] = final_value_loss

        # ── Phase 3: Battle – Neues vs. Bestes Modell ──
        is_best = False
        if (iteration + 1) % 20 == 0:
            print("  Starte Battle-Phase gegen bisheriges bestes Modell...")
            mcts_new = mcts # mcts nutzt bereits das iterativ weiterlernende model
            
            model_old.set_weights(best_weights)

            new_wins = 0
            old_wins = 0
            draws = 0

            # Wir spielen N_BATTLE Spiele (N_BATTLE/2 Paarungen)
            for g in range(N_BATTLE // 2):
                for queen_color in [chess.WHITE, chess.BLACK]:
                    # Ein Board pro Paarung generieren
                    board, _, _, _ = generate_random_endgame(tablebase, max_dtz, queen_color, endgame_dict)
                    
                    # Spiel 1: Neu=Weiß, Alt=Schwarz
                    result = battle_game(mcts_new, mcts_old, board)
                    if result == '1-0':
                        new_wins += 1
                    elif result == '0-1':
                        old_wins += 1
                    else:
                        draws += 1
                    
                    # Spiel 2: Alt=Weiß, Neu=Schwarz (auf dem exakt selben Board)
                    result = battle_game(mcts_old, mcts_new, board)
                    if result == '1-0':
                        old_wins += 1
                    elif result == '0-1':
                        new_wins += 1
                    else:
                        draws += 1

            print(f"Battle: Neu {new_wins} – Remis {draws} – Alt {old_wins}")
            epoch_metrics['battle_new_wins'] = new_wins
            epoch_metrics['battle_old_wins'] = old_wins
            epoch_metrics['battle_draws'] = draws

            if new_wins > old_wins:
                print(f"Neues Modell ist stärker! (Gewichte als Best markiert)")
                best_weights = model.get_weights()
                is_best = True
            else:
                print(f"Neues Modell verliert. Training läuft trotzdem mit neuen Gewichten weiter!")

        # ── Phase 4: Curriculum Aufstieg prüfen ──
        if len(recent_results) >= 20 and win_threshold is not None:
            queen_wins = sum(1 for r in recent_results if r)
            win_rate = queen_wins / len(recent_results)
            epoch_metrics['win_rate'] = win_rate
            print(f"Win-Rate ({primary_type}): {win_rate:.1%} (Threshold: {win_threshold:.0%})")
            if win_rate >= win_threshold and phase < len(CURRICULUM) - 1:
                phase += 1
                max_dtz, win_threshold, endgame_dict = CURRICULUM[phase]
                recent_results.clear()
                epoch_metrics['upgraded_phase'] = True
                print(f"Aufstieg zu Phase {phase + 1}! Neues Endgame: {endgame_dict}, DTZ={max_dtz}")
                # LR Schedule: Feinere Lernrate für schwierigere Phasen
                if phase >= 3:
                    new_lr = 1e-4
                elif phase >= 2:
                    new_lr = 5e-4
                else:
                    new_lr = 1e-3
                model.optimizer.learning_rate.assign(new_lr)
                print(f"  LR angepasst auf {new_lr}")
        else:
            queen_wins = sum(1 for r in recent_results if r)
            epoch_metrics['win_rate'] = queen_wins / max(1, len(recent_results))

        # Checkpoint speichern
        log_metrics(epoch_metrics)
        save_checkpoint(model, iteration, phase, recent_results, is_best=is_best)


if __name__ == '__main__':
    train()
