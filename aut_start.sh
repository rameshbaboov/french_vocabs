#!/bin/bash
set -x
cd /home/rameshbaboov/rameshbaboov/aiprojects/french_vocabs

# directly use full path to venv python, no need for "source"
PYTHON="/home/rameshbaboov/rameshbaboov/aiprojects/french_vocabs/venv/bin/python"

tmux new-session -d -s A1_wordgen "$PYTHON generate_words.py --level=A1 --model=gemma2:2b"
tmux new-session -d -s A2_wordgen "$PYTHON generate_words.py --level=A2 --model=gemma2:2b"
tmux new-session -d -s B1_wordgen "$PYTHON generate_words.py --level=B1 --model=gemma2:2b"
tmux new-session -d -s B2_wordgen "$PYTHON generate_words.py --level=B2 --model=gemma2:2b"
tmux new-session -d -s A1_Sengen "$PYTHON generate_sentences.py --level=A1 --model=gemma2:2b --output-dir=A1OUT"
tmux new-session -d -s A2_Sengen "$PYTHON generate_sentences.py --level=A2 --model=gemma2:2b --output-dir=A2OUT"
tmux new-session -d -s B1_Sengen "$PYTHON generate_sentences.py --level=B1 --model=gemma2:2b --output-dir=B1OUT"
tmux new-session -d -s B2_Sengen "$PYTHON generate_sentences.py --level=B2 --model=gemma2:2b"


echo "started tmux sessions"
