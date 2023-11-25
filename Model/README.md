# Model Training

This folder contain three python files:
- `trainer.py`: a training driver that is easy to train a model and save them
- `test.py`: a predicting driver that is easy to inference labels, it's also the entry point that inference the answer
- `train`.py: entry point, which trains models

## Hyperparameters
We adpat ensemble trick, so there are three base *CatBoostClassifier* models.

Each is specified in `config0.json`, `config1.json`, `config2.json`
