import json
import os
import numpy as np
from trainer import Trainer
from test import Predictor

SEED = 42
np.random.seed(SEED)

TESTID = 22
scope = "train60"
config = {
    "seed": SEED,
    "model_kwargs": {
        "random_seed": SEED,
        "task_type": "GPU",
        "devices": [0, 1],
        "auto_class_weights": "SqrtBalanced",
        "eval_metric": "F1",
        "iterations": 1000,
        "depth": 12,
        "l2_leaf_reg": 4,
        "bootstrap_type": "Poisson",
    },
    "fit_kwargs": {"verbose": 200},
}
# save config to config.json in model_path
os.makedirs(f"model/{scope}/{TESTID}", exist_ok=True)
with open(f"model/{scope}/{TESTID}/config.json", "w") as f:
    json.dump(config, f)


if __name__ == "__main__":
    # # training
    trainer = Trainer({"model_path": f"model/{scope}/{TESTID}", "scope": scope})
    trainer.train(
        {
            "model_config": config["model_kwargs"],
            "fit_config": config["fit_kwargs"],
        }
    )

    # load model
    model_solo, model_multi = trainer.load_model(f"model/{scope}/{TESTID}", export=True)

    print("start predicting...")
    predictor = Predictor(model_solo, model_multi)

    root = "data/history"
    # eval
    preds = predictor.inference([f"{root}/val", f"{root}/public"], mode="val")

    exit()
