# dermamnist-cv-messy
another medmnist project. this one is skin lesions (DermaMNIST), 7 classes.
downloads data on demand so nothing is committed in /data.
not cleaned, just "works".
## run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m src.train.train1
python -m src.train.eval_it
output: runs/best.pt and some printouts