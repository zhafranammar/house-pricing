"""Jalankan Flask secara lokal dari root proyek."""

import os
import sys

APP_DIR = os.path.join(os.path.dirname(__file__), "app")
sys.path.insert(0, APP_DIR)

from __init__ import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)
