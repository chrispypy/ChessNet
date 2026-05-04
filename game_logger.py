"""
GameLogger: Schreibt den aktuellen Spielstand als JSON raus,
damit der Viewer ihn live anzeigen kann.

Einbindung in selfplay.py:
    logger = GameLogger()
    # Nach jedem Zug:
    logger.log_move(board, move)
    # Bei Spielende:
    logger.game_over(board)
"""

import json
import os
import chess

GAME_LOG_PATH = os.path.join(os.path.dirname(__file__), 'current_game.json')


MAX_HISTORY = 20  # Anzahl fertige Spiele die gespeichert werden


class GameLogger:
    def __init__(self, path=GAME_LOG_PATH, is_logging=True):
        self.path = path
        self.is_logging = is_logging
        self.moves = []
        self.fens = []
        self.evals = []  # Netz-Eval pro Zug
        self.game_number = 0
        self.stats = {'white_wins': 0, 'black_wins': 0, 'draws': 0}
        self.finished_games = []

    def new_game(self, board: chess.Board):
        """Neues Spiel starten."""
        self.game_number += 1
        self.moves = []
        self.fens = [board.fen()]
        self.evals = []
        self._write()

    def log_move(self, board: chess.Board, move: chess.Move, eval_score=None):
        """Zug loggen (nach board.push). eval_score: P(Win)-P(Loss) aus Sicht Weiß."""
        self.moves.append(move.uci())
        self.fens.append(board.fen())
        if eval_score is not None:
            self.evals.append(round(float(eval_score), 3))
        # Nur alle 5 Züge auf Disk schreiben (game_over schreibt immer)
        if len(self.moves) % 5 == 0:
            self._write()

    def game_over(self, board: chess.Board, result: str):
        """Spielende markieren und Statistik updaten."""
        if result == '1-0':
            self.stats['white_wins'] += 1
        elif result == '0-1':
            self.stats['black_wins'] += 1
        else:
            self.stats['draws'] += 1

        game = {
            'game_number': self.game_number,
            'fens': list(self.fens),
            'moves': list(self.moves),
            'evals': list(self.evals),
            'result': result,
        }
        self.finished_games.append(game)
        if len(self.finished_games) > MAX_HISTORY:
            self.finished_games.pop(0)

        self._write(result=result)

    def _write(self, result=None):
        if not self.is_logging:
            return
        data = {
            'game_number': self.game_number,
            'fens': self.fens,
            'moves': self.moves,
            'move_count': len(self.moves),
            'result': result,
            'running': result is None,
            'stats': self.stats,
            'finished_games': self.finished_games,
        }
        with open(self.path, 'w') as f:
            json.dump(data, f)
