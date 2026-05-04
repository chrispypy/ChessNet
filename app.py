import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import chess
import numpy as np
import keras
import time
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from encoder import Encoder, BoardEncoder
from gui_mcts import GuiMCTS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
model = None
encoder = None
board_encoder = None
mcts_worker = None
mcts_thread = None

# The network only supports Queen/Rook/King moves. Set a valid KQvKR FEN!
current_board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 w - - 0 1")
max_simulations = 1000 # Default to 1000 sims instead of infinite
engine_enabled = True

def mcts_emit_callback(root, sims_done, root_eval):
    moves_data = []
    for i, move in enumerate(root.legal_moves):
        # Calculate Q-Value based on our new Minimax convention
        if move in root.children:
            q = -root.children[move].q_value
        else:
            q = 0.0
            
        if move in root.fatal_moves:
            q = -1.0 # Display -1.0 for fatal
            
        moves_data.append({
            'san': root.board.san(move),
            'uci': move.uci(),
            'visits': int(root.child_visits[i]),
            'q': float(q),
            'prior': float(root.child_priors[i])
        })
        
    moves_data.sort(key=lambda x: x['visits'], reverse=True)
    
    socketio.emit('mcts_update', {
        'sims_done': sims_done,
        'root_eval': float(root_eval),
        'moves': moves_data
    })

def run_mcts_loop():
    global mcts_worker, current_board, max_simulations, engine_enabled
    while True:
        if mcts_worker and engine_enabled:
            try:
                mcts_worker.start_continuous_search(current_board, max_simulations, mcts_emit_callback)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"MCTS error: {e}")
        time.sleep(0.1)

def re_start_search():
    global mcts_worker
    if mcts_worker:
        mcts_worker.stop_flag = True
        time.sleep(0.05) # wait for loop to restart

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    print("Client connected")
    socketio.emit('board_state', {'fen': current_board.fen()})

redo_stack = []

@socketio.on('set_fen')
def set_fen(data):
    global current_board, redo_stack
    fen = data.get('fen')
    try:
        current_board = chess.Board(fen)
        redo_stack.clear()
        socketio.emit('board_state', {'fen': current_board.fen()})
        re_start_search()
    except Exception as e:
        print("Invalid FEN", e)

@socketio.on('make_move')
def make_move(data):
    global current_board, redo_stack
    uci = data.get('uci')
    try:
        move = chess.Move.from_uci(uci)
        if move in current_board.legal_moves:
            current_board.push(move)
            redo_stack.clear()
            socketio.emit('board_state', {'fen': current_board.fen()})
            re_start_search()
    except Exception as e:
        print("Invalid Move", e)

@socketio.on('undo_move')
def undo_move():
    global current_board, redo_stack
    if current_board.move_stack:
        move = current_board.pop()
        redo_stack.append(move)
        socketio.emit('board_state', {'fen': current_board.fen()})
        re_start_search()

@socketio.on('redo_move')
def redo_move():
    global current_board, redo_stack
    if redo_stack:
        move = redo_stack.pop()
        current_board.push(move)
        socketio.emit('board_state', {'fen': current_board.fen()})
        re_start_search()

@socketio.on('set_max_sims')
def set_max_sims(data):
    global max_simulations
    max_simulations = int(data.get('max_sims', 0))
    re_start_search()

@socketio.on('set_engine_state')
def set_engine_state(data):
    global engine_enabled, mcts_worker
    engine_enabled = data.get('enabled', True)
    if not engine_enabled and mcts_worker:
        mcts_worker.stop_flag = True
    else:
        re_start_search()

def init():
    global model, encoder, board_encoder, mcts_worker
    print("Loading model...")
    model_path = os.path.join('checkpoints', 'model_best.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join('checkpoints', 'model_latest.keras')
    model = keras.models.load_model(model_path)
    encoder = Encoder()
    board_encoder = BoardEncoder()
    mcts_worker = GuiMCTS(model, encoder, board_encoder)
    print("Model loaded. Server ready at http://127.0.0.1:5000")
    
    # Start background loop
    threading.Thread(target=run_mcts_loop, daemon=True).start()

init()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
