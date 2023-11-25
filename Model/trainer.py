"""
This code is doing:
1. Define a Trainer class
2. Easy to train a model and save it
3. Easy to load a model
"""
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


class Trainer:
    """
    This class is used to train a model.

    Attributes:
        - models: a dict of models (solo and multi)
        - config: a dict of config
    Methods:
        train: train a model
        predict: predict a model
        save: save a model
        load: load a model
    """

    def __init__(self, config: dict) -> None:
        """
        config: {'model_path': ...,}
        return: None
        """
        self.models = {"solo": None, "multi": None}
        self.config = config

        # Check if model_path exists in config
        if "model_path" not in self.config:
            raise ValueError("model_path not found in config")

    def _load_train_data(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """
        mode: "solo" or "multi"
        return: train_X, train_y
        """
        if mode == "solo":
            path = f"./data/history/{self.config['scope']}/train_1.csv"
        elif mode == "multi":
            path = f"./data/history/{self.config['scope']}/train_multi.csv"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        train_data = pd.read_csv(path, engine="pyarrow")
        # train_data.drop(columns=["txkey"], inplace=True)

        train_X = train_data.drop(columns=["label"])
        train_y = train_data["label"]

        return train_X, train_y

    def train(self, config: dict = {}) -> None:
        """
        config: training config

        config:
            e.g. {'solo': _config, 'multi': _config}
            e.g. {_config}
        _config: {'model_type': ..., 'model_config': ..., fit_config: ...}
        """

        # Set default config
        configs = {}
        configs["solo"] = config.get("solo", None)
        configs["multi"] = config.get("multi", None)
        if configs["solo"] is None and configs["multi"] is None:
            configs["solo"] = configs["multi"] = config

        # Train solo and multi models
        for mode, config in configs.items():
            if config is not None:
                # Load data
                train_X, train_y = self._load_train_data(mode)

                # Create model
                self.models[mode] = {}
                if config.get("model_type", "cat") == "cat":
                    self.models[mode]["types"] = "cat"
                    self.models[mode]["model"] = CatBoostClassifier(**config.get("model_config", {}))
                else:
                    self.models[mode]["types"] = "xgb"
                    self.models[mode]["model"] = XGBClassifier(**config.get("model_config", {}))

                # Train model
                print(f"Training {mode} model")
                print(f'Random state: {self.models[mode]["model"].get_param("random_seed")}')
                self.models[mode]["model"].fit(train_X, train_y, **config.get("fit_config", {}))

                # Save model
                self._save(mode)

    def _save(self, mode: str) -> None:
        """
        Save model

        mode: "solo" or "multi"
        return: None, but save model
        """
        if os.path.exists(self.config["model_path"]) is False:
            os.makedirs(self.config["model_path"])
        filename = f'{self.config["model_path"]}/{mode}.model'
        self.models[mode]["model"].save_model(filename)
        print(f"Saved: {filename}")

    def load_model(self, path: str, config: str | dict = "cat", export=False) -> None:
        """
        Load model according to config

        path: model path
        config: model config
            e.g. {'solo': 'cat', 'multi': 'xgb'} or 'cat'
        """
        # Set default config
        configs = {}
        if isinstance(config, str):
            configs["solo"] = configs["multi"] = config
        elif isinstance(config, dict):
            configs["solo"] = config.get("solo", None)
            configs["multi"] = config.get("multi", None)
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Load solo and multi models
        for mode, config in configs.items():
            if config is not None:
                self.models[mode] = {}
                # Load model
                if config == "cat":
                    self.models[mode]["types"] = "cat"
                    self.models[mode]["model"] = CatBoostClassifier()
                elif config == "xgb":
                    self.models[mode]["types"] = "xgb"
                    self.models[mode]["model"] = XGBClassifier()

                filename = f"{path}/{mode}.model"
                self.models[mode]["model"].load_model(filename)
                print(f"Loaded {mode} model: {filename}")

        # Alarm if not all models are loaded
        if self.models["solo"] is None or self.models["multi"] is None:
            print("Warning: Not all models are loaded")

        if export:
            models = [self.models[key]["model"] for key in self.models]
            return models


if __name__ == "__main__":
    trainer = Trainer({"model_path": "mm"})
    trainer.train(config={"solo": {"model_type": "cat", "fit_config": {"verbose": 200}}})
    trainer.load("mm", {"solo": "cat"})
