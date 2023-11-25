"""
This file is used to train the model with trainer.
Input: preprocessed data
Output: trained model
"""
import json

import numpy as np
from trainer import Trainer

SEED = 42
np.random.seed(SEED)
scope = "train60"

if __name__ == "__main__":
    # training
    for testid in range(3):
        # load config
        with open(f"./Model/config{testid}.json") as json_file:
            config = json.load(json_file)

        # output path: ["./model/train60/20", "./model/train60/21", "./model/train60/22"]
        trainer = Trainer({"model_path": f"model/{scope}/{20+testid}", "scope": scope})
        trainer.train(
            {
                "model_config": config["model_kwargs"],
                "fit_config": config["fit_kwargs"],
            }
        )

    exit()
