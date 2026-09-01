#!/bin/bash
# Full paper build: regenerate tables/numbers from results, compile (pdfTeX,
# TinyTeX), then run the mechanical checks. Fails on the first error.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"

$PY scripts/make_tables.py

cd paper
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex > build/pass1.out \
  || { tail -30 build/pass1.out; exit 1; }
bibtex build/main > build/bibtex.out || { cat build/bibtex.out; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex > build/pass3.out \
  || { tail -30 build/pass3.out; exit 1; }
cd ..

$PY scripts/make_tables.py --check
$PY scripts/check_paper.py
echo "BUILD OK: paper/build/main.pdf"
