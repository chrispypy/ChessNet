## Usage

### 1. Testing the pre-trained model
The `/checkpoints` directory already contains the currently strongest pre-trained model (`model_best.keras`). You can use this out-of-the-box to play directly against the engine or test the network's performance without investing your own compute time into self-play.

### 2. Starting your own training (Self-Play)
To start the learning process from scratch and train the neural network via self-play, simply run the following command:

    python selfplay.py

Note on starting positions: The training uses a curriculum learning approach. To generate meaningful and solvable endgames (like KQvK or KRvK) at the beginning, the script relies on 3- and 4-piece Syzygy tablebases. These are already fully included in the `/syzygy` directory, so you don't need to download anything else. All required output directories (e.g., for new checkpoints) are created automatically.

### 3. Live Viewer: Watching the learning process
The project includes a custom local web dashboard to visualize the training progress and MCTS in real-time. To use it, keep `selfplay.py` running and start the server in a second terminal window:

    python serve_viewer.py

Then, simply open `http://localhost:8080` in your browser. You will see the current board states, move history, and the network's evaluation (win/loss probability) live.

### 4. Important note on evaluation (Best Model Tracking)
The training script includes an automated battle mode: every few epochs, the newly trained model plays against the current best model. Only if the new model wins is it saved as the new `model_best.keras`.

If you want to start your training completely from scratch (tabula rasa):
Please rename or delete the provided model in the `/checkpoints` directory beforehand. Why? Since the included model is already very strong, your new, untrained network will lose every evaluation game initially. It would take a disproportionately long time for your network to reach this high level and beat the internal benchmark.
