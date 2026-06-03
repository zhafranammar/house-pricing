"""WSGI entry point untuk gunicorn (dijalankan dari folder app/)."""

from __init__ import create_app

app = create_app()
