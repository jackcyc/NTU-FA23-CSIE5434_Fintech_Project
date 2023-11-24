import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from preprocess import load_data, save_data
from trainer import Trainer

SEED = 42
np.random.seed(SEED)


def sentinel(path):
    # load csv
    pred = pd.read_csv(path, engine="pyarrow")
    private = pd.read_csv("./data/dataset/private_1_processed.csv", engine="pyarrow")
    public = pd.read_csv("./data/dataset/public.csv", engine="pyarrow")

    # Check pred has all txkey contained in private
    assert len(set(private["txkey"]) - set(pred["txkey"])) == 0

    # Check 0, 1 ratio
    print("Sentinel:")
    pred.set_index("txkey", inplace=True)
    df = pd.DataFrame()
    df["public"] = [public["label"].value_counts(normalize=True)[1]]
    df["pred_private"] = [pred.loc[private["txkey"]].value_counts(normalize=True)[1]]
    df["pred_public"] = [
        pred.loc[list(set(pred.index) - set(private["txkey"]))].value_counts(
            normalize=True
        )[1]
    ]
    print(df.head())

    if df["pred_private"].item() >= 0.005 or df["pred_private"].item() <= 0.0025:
        raise ValueError("Wierd 1 ratio in pred_private")
    if df["pred_public"].item() >= 0.005 or df["pred_public"].item() <= 0.0025:
        raise ValueError("Wierd 1 ratio in pred_public")
    print("Passed!")


class Predictor:
    def __init__(self, paths: list) -> None:
        self.model_solo = []
        self.model_multi = []
        for path in paths:
            solo, multi = Trainer({"model_path": "model"}).load_model(
                f"{path}", export=True
            )
            self.model_solo.append(solo)
            self.model_multi.append(multi)

    def load_data(self, dir: str) -> None:
        name = os.path.basename(dir)
        assert name in {"val", "public", "private1"}
        # load data
        hist, nohist1, nohisto = load_data(
            root=dir,
            names=["have_history", "no_history_first", "no_history_others"],
        )

        # gt
        hist_gt = hist[["txkey", "label"]]
        nohist1_gt = nohist1[["txkey", "label"]]
        nohisto_gt = nohisto[["txkey", "label"]]
        # inputs for model
        hist = hist.drop(columns=["label"])
        nohist1 = nohist1.drop(columns=["label"])
        nohisto = nohisto.drop(columns=["label"])

        self.input = (hist, nohist1, nohisto)
        self.gt = (hist_gt, nohist1_gt, nohisto_gt)

    def make_evaluatoin(self, pred: pd.DataFrame) -> None:
        hist_gt, nohist1_gt, nohisto_gt = self.gt
        hist_pred_gt = hist_gt.merge(pred, on="txkey", how="left", validate="1:1")
        nohist1_pred_gt = nohist1_gt.merge(pred, on="txkey", how="left", validate="1:1")
        nohisto_pred_gt = nohisto_gt.merge(pred, on="txkey", how="left", validate="1:1")

        all_pred_gt = pd.concat(
            [hist_pred_gt, nohist1_pred_gt, nohisto_pred_gt], axis=0
        )

        self.show_result(
            {
                "all": all_pred_gt,
                "hist": hist_pred_gt,
                "nohist1": nohist1_pred_gt,
                "nohisto": nohisto_pred_gt,
            }
        )

    def make_prediction(
        self, hist: pd.DataFrame, nohist1: pd.DataFrame, nohisto: pd.DataFrame
    ) -> pd.DataFrame:
        hist = hist[["txkey", "pred"]]
        nohist1 = nohist1[["txkey", "pred"]]
        nohisto = nohisto[["txkey", "pred"]]
        data = pd.concat([hist, nohist1, nohisto], axis=0)
        return data

    def baseline(
        self,
        method: str,
        hist: pd.DataFrame,
        nohist1: pd.DataFrame,
        nohisto: pd.DataFrame,
    ) -> pd.DataFrame:
        assert method in {"solo", "history", "history_solo"}
        solo_to_drop = ["txkey", "cano", "label"]
        multi_to_drop = ["txkey", "cano", "label", "time_diff", "fraudrate"]

        if method == "solo":
            hist["pred"] = self.model_solo.predict(hist.drop(columns=multi_to_drop))
            nohist1["pred"] = self.model_solo.predict(
                nohist1.drop(columns=solo_to_drop)
            )
            nohisto["pred"] = self.model_solo.predict(
                nohisto.drop(columns=multi_to_drop)
            )
        else:  # history, history_solo
            cano2prevlabel = hist.groupby("cano")["prev_label"].first()
            hist["pred"] = hist["cano"].apply(lambda x: cano2prevlabel[x])
            if method == "history":
                nohist1["pred"] = 0
                nohisto["pred"] = 0
            elif method == "history_solo":
                nohist1["pred"] = self.model_solo.predict(
                    nohist1.drop(columns=solo_to_drop)
                )
                nohisto["pred"] = self.model_solo.predict(
                    nohisto.drop(columns=multi_to_drop)
                )
        return self.make_prediction(hist, nohist1, nohisto)

    def predict_proba(self, model_type: str, data: pd.DataFrame):
        """
        Ensemble version of predict_proba
        """
        assert model_type in {"solo", "multi"}
        preds = []
        if model_type == "solo":
            for model in self.model_solo:
                preds.append(model.predict_proba(data)[:, 1])
        else:
            for model in self.model_multi:
                preds.append(model.predict_proba(data)[:, 1])
        preds = np.mean(preds, axis=0)
        return preds

    def foward_multi(self, data: pd.DataFrame):
        """
        Input should contain:
            ['loctm', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
            'ovrlt', 'csmam', 'flg_3dsmk', 'contp_0', 'contp_1', 'contp_2',
            'contp_3', 'etymd_0', 'etymd_1', 'etymd_2', 'etymd_3', 'hcefg_0',
            'hcefg_1', 'hcefg_2', 'hcefg_3', 'mcc_0', 'mcc_1', 'mcc_2', 'stocn_0',
            'stocn_1', 'stocn_2', 'scity_0', 'scity_1', 'scity_2', 'csmcu_0',
            'csmcu_1', 'csmcu_2', 'time_diff', 'fraudrate', 'prev_label']
        Predict both prev0 and prev1
        """
        data["prev_label"] = 0
        data["prev_0_prob"] = self.predict_proba(
            "multi", data.drop(columns=["txkey", "cano"])
        )
        # data["prev_0_prob"] = (data["prev_0_prob"] + data["prob"]) / 2
        data["prev_0_prob"] = data["prev_0_prob"].apply(lambda x: 1 if x > 0.5 else 0)

        data["prev_label"] = 1
        data["prev_1_prob"] = self.predict_proba(
            "multi", data.drop(columns=["txkey", "cano"])
        )
        # data["prev_1_prob"] = (data["prev_1_prob"] + data["prob"]) / 2
        data["prev_1_prob"] = data["prev_1_prob"].apply(lambda x: 1 if x > 0.5 else 0)
        return data

    def forward(self, hist: pd.DataFrame, nohist1: pd.DataFrame, nohisto: pd.DataFrame):
        # 1. model_solo
        nohist1["prob"] = self.predict_proba(
            "solo", nohist1.drop(columns=["txkey", "cano"])
        )
        nohist1["pred"] = nohist1["prob"].apply(lambda x: 1 if x > 0.5 else 0)

        nohisto["prob"] = self.predict_proba(
            "solo", nohisto.drop(columns=["txkey", "cano"])
        )
        nohisto["pred"] = nohisto["prob"].apply(lambda x: 1 if x > 0.5 else 0)

        cano2fraudrate = (
            pd.concat([nohist1, nohisto], axis=0).groupby("cano")["pred"].mean()
        )
        nohisto["fraudrate"] = nohisto["cano"].apply(lambda x: cano2fraudrate[x])

        cano2prev = pd.concat(
            [
                hist.groupby("cano")["prev_label"].first(),
                nohist1.groupby("cano")["pred"].first(),
            ],
            axis=0,
        )

        # model_multi, predict both prev0 and prev1
        nohisto = self.foward_multi(nohisto)
        hist = self.foward_multi(hist)

        # autoreregressive
        def autoregressive(array, prev_label):
            labels = np.zeros(len(array), dtype=int)
            for i in range(len(array)):
                label = array[i, prev_label]
                labels[i] = prev_label = label
            return labels

        print("start autoregression...")
        start_t = time.time()
        preds = []
        for cano, data in nohisto.groupby("cano"):
            array = data[["prev_0_prob", "prev_1_prob"]].values
            # prev_labels = cano2prev[cano]
            pred = autoregressive(array, prev_label=cano2prev[cano])
            preds.append(pred)
        print("end autoregression...", time.time() - start_t)
        preds = np.concatenate(preds)
        nohisto["pred"] = preds

        print("start autoregression...")
        start_t = time.time()
        preds = []
        for cano, data in hist.groupby("cano"):
            array = data[["prev_0_prob", "prev_1_prob"]].values
            # prev_labels.append(cano2prev[cano])
            pred = autoregressive(array, prev_label=cano2prev[cano])
            preds.append(pred)
        # start = time.time()
        # preds = pool.map(autoregressive, arrays, prev_labels)
        print("end autoregression...", time.time() - start_t)
        preds = np.concatenate(preds)
        hist["pred"] = preds

        return self.make_prediction(hist, nohist1, nohisto)

    def show_result(self, preds: dict) -> None:
        # show brief f1 score
        print("\n" + "-" * 20)
        print("name\tf1 score\tprecision\trecall")
        for name, pred in preds.items():
            print(
                name,
                f"{f1_score(pred['label'], pred['pred']):.3f}",
                f"{precision_score(pred['label'], pred['pred']):.3f}",
                f"{recall_score(pred['label'], pred['pred']):.3f}",
                sep="\t",
            )

    def inference(self, paths: list, mode: str):
        assert mode in {"val", "test"}
        if isinstance(paths, str):
            paths = [paths]

        preds = []
        for path in paths:
            # load data to self.input
            self.load_data(path)
            # predict result from self.input
            hist, nohist1, nohisto = self.input
            pred = self.forward(hist, nohist1, nohisto)
            preds.append(pred)
            if mode == "val":
                # make evaluation
                self.make_evaluatoin(pred)
        preds = pd.concat(preds, axis=0)

        if mode == "test":
            sample = load_data("data/misc", "31_範例繳交檔案")
            sampel_txkeys = set(sample["txkey"])

            preds = preds[preds["txkey"].isin(sampel_txkeys)]
            print(f"len(preds): {len(preds)}, len(sampel_txkeys): {len(sampel_txkeys)}")

        return preds


if __name__ == "__main__":
    # load model
    print("Loading model...")
    predictor = Predictor(["model/train60/20", "model/train60/21", "model/train60/22"])

    root = "data/history"
    # eval
    # preds = predictor.inference([f"{root}/val", f"{root}/public"], mode="val")

    # test
    output_filename = "tmp"
    preds = predictor.inference([f"{root}/private1", f"{root}/public"], mode="test")
    save_data("result", {output_filename: preds})
    sentinel(f"result/{output_filename}.csv")

    exit()
