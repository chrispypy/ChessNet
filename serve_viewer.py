"""
Startet einen lokalen Server für den Live-Viewer.

Usage:
    python serve_viewer.py

Dann im Browser: http://localhost:8080
"""

import http.server
import os

PORT = 8080
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        # Kein Caching für JSON (damit Polling funktioniert)
        if self.path.endswith('.json'):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, format, *args):
        # Nur Nicht-JSON-Requests loggen (sonst spammt das Polling)
        if '.json' not in str(args[0]):
            super().log_message(format, *args)


if __name__ == '__main__':
    print(f"Viewer läuft auf http://localhost:{PORT}/viewer.html")
    server = http.server.HTTPServer(('', PORT), Handler)
    server.serve_forever()
