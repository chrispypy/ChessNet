## Usage / Verwendung

### 1. Das vortrainierte Modell testen
Im Verzeichnis `/checkpoints` liegt bereits das aktuell stärkste, vortrainierte Modell (`model_best.keras`). Du kannst dieses "Out-of-the-Box" verwenden, um direkt gegen die Engine zu spielen oder die Performance des Netzwerks zu testen, ohne erst eigene Rechenzeit in das Self-Play investieren zu müssen.

### 2. Eigenes Training (Self-Play) starten
Um den Lernprozess von Grund auf neu zu starten und das neuronale Netz durch Self-Play trainieren zu lassen, genügt folgender Befehl:

    python selfplay.py

Hinweis zu den Startstellungen: Das Training nutzt einen Curriculum-Learning-Ansatz. Um zu Beginn sinnvolle und lösbare Endspiele (wie KQvK oder KRvK) zu generieren, greift das Skript auf 3- und 4-Steiner Syzygy-Tablebases zurück. Diese sind im Ordner `/syzygy` bereits vollständig enthalten, du musst also nichts weiter herunterladen. Alle benötigten Output-Verzeichnisse (z. B. für neue Checkpoints) werden automatisch erstellt.

### 3. Live-Viewer: Dem Lernprozess zuschauen
Das Projekt beinhaltet ein eigenes, lokales Web-Dashboard, um den Trainingsfortschritt und das MCTS in Echtzeit zu visualisieren. Lass dazu `selfplay.py` laufen und starte in einem zweiten Terminal-Fenster den Server:

    python serve_viewer.py

Öffne anschließend einfach `http://localhost:8080` in deinem Browser. Du siehst dort live die aktuellen Brettstellungen, die Zug-Historie und die Evaluation (Win/Loss-Probability) des neuronalen Netzes.

### 4. Wichtiger Hinweis zur Evaluierung (Best Model Tracking)
Das Trainingsskript beinhaltet einen automatisierten Battle-Modus: Alle paar Epochen tritt das neu trainierte Modell gegen das bisher beste Modell an. Nur wenn das neue Modell gewinnt, wird es als neues `model_best.keras` gespeichert.

Wenn du dein Training komplett bei Null (Tabula Rasa) beginnen möchtest:
Bitte benenne vorher das mitgelieferte Modell im Ordner `/checkpoints` um oder lösche es. Warum? Da das mitgelieferte Modell bereits sehr stark ist, wird dein neues, untrainiertes Netz am Anfang jedes Evaluierungsspiels verlieren. Es würde unverhältnismäßig lange dauern, bis dein Netz dieses hohe Niveau erreicht hat und den internen Benchmark schlägt.
