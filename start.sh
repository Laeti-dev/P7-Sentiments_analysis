#!/bin/bash

# Lancer Flask avec Gunicorn en arrière-plan
gunicorn --bind 0.0.0.0:8080 app:app &

# Lancer Streamlit sur le port 8501
streamlit run your_streamlit_app.py --server.port 8501 --server.address 0.0.0.0
