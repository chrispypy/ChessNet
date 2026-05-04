## Usage / Verwendung

### Das vortrainierte Modell ausprobieren
Im Ordner `/checkpoints` liegt bereits mein bisher bestes Modell. Du kannst es direkt verwenden, um die Leistung des Bots auszuprobieren, ohne selbst trainieren zu müssen.

### Ein eigenes Modell trainieren
Um das Training (Self-Play) von Grund auf neu zu starten, führe einfach dieses Skript aus:

```bash
python selfplay.py
```
In dem Ordner syzygy sind die benötigten Tablebases, um die startstellungen für das Training zu generieren.
*Hinweis: Alle benötigten Dateien und Ordner (wie `/checkpoints`) werden dabei automatisch angelegt.*

### Live beim Training zuschauen
Während `selfplay.py` läuft, kannst du dem Bot in Echtzeit beim Spielen und Lernen zuschauen. Starte dazu einfach in einem zweiten Terminal-Fenster den Viewer:

```bash
python serve_viewer.py
```

### Wichtiger Hinweis zum "Best Model Tracking"
Wenn du ein eigenes Modell trainierst und vom System tracken lassen möchtest, wann es ein neues "bestes Modell" gibt, **solltest du mein mitgeliefertes Modell in `/checkpoints` vorher umbenennen oder löschen.** *Warum?* Das mitgelieferte Modell ist bereits sehr stark. Wenn du es im Ordner lässt, muss dein neu trainiertes Modell erst dieses hohe Niveau erreichen und es schlagen, bevor es als neues bestes Modell gespeichert wird – was unter Umständen sehr lange dauern kann.
