"""
This file is used to preprocess the data.
Input: raw data to be preprocessed
Output: preprocessed data (data/history/*)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, save_data


def preprocess_base(data, is_train):
    """
    Simple preprocess to reduce file size

    Args:
        data (pandas.DataFrame): The input data to be preprocessed.
        is_train (bool): A flag indicating whether the data is for training or not.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    if is_train:
        # 1. turn txkey into serial number
        data.reset_index(drop=True, inplace=True)
        data["txkey"] = data.index
        # 2. drop duplicated rows? but maybe it's due to continuous transaction in a short time(<1s)
        # data.drop_duplicates(inplace=True)
    # 3. chid, cano, mchno can be shorten to last 10 digits. acqic ~ 7 digits
    data["chid"] = data["chid"].apply(lambda x: x[-10:])
    data["cano"] = data["cano"].apply(lambda x: x[-10:])
    data["mchno"] = data["mchno"].apply(lambda x: x[-10:])
    data["acqic"] = data["acqic"].apply(lambda x: x[-7:])
    # 4. stscd ~100%=NaN, flbmk ~100%=0
    data.drop(columns=["stscd", "flbmk"], inplace=True)
    return data


def preprocess_xgb(data):
    """
    Preprocesses the given data for XGBoost-like model.

    Args:
        data (pandas.DataFrame): The input data to be preprocessed.

    Returns:
        pandas.DataFrame: The preprocessed data.

    Notes:
        - All attributes will be numerical or boolean (one-hot encoded).
        - Special attributes: txkey, cano, abs_time, label.
        - Drops columns: chid, mchno, acqic.
        - Converts loctm to abs_time and loctm(secs).
        - Converts contp, etymd, iterm, hcefg to categorical variables.
        - Preprocesses sparse categorical variables: mcc, stocn, scity, csmcu.
        - Sorts the data by cano and abs_time.
    """
    # drop chid, mchno, acqic
    data.drop(columns=["chid", "mchno", "acqic"], inplace=True)
    # locdt, loctm -> abs_time, loctm_secs
    data["loctm"] = data["loctm"].apply(
        lambda x: ((x // 10000) * 3600 + (x // 100 % 100) * 60 + x % 100) / 86400
    )
    data["abs_time"] = data["locdt"] + data["loctm"]  # e.g. 3rd day noon = 3.50000
    data.drop(columns=["locdt"], inplace=True)
    # bool: ['ecfg', 'insfg', 'bnsfg', 'ovrlt', 'flg_3dsmk']
    # numerical: median = ~300
    num_cols = ["conam", "flam1", "csmam"]
    for col in num_cols:
        data[col] = np.log(data[col].fillna(300) + 1) / 10
    data["flam1"] = data["flam1"] - data["conam"]  # final - original

    # cate: ['contp', 'etymd', 'iterm', 'hcefg']
    # contp -> 4 cates: 5, 4, 2, others
    data["contp"] = data["contp"].apply(
        lambda x: 0 if x == 5 else 1 if x == 4 else 2 if x == 2 else 3
    )
    # etymd -> 4, (5,8,1), 0, other
    data["etymd"] = data["etymd"].apply(
        lambda x: 0 if x == 0 else 1 if x in [5, 8, 1] else 2 if x == 4 else 3
    )
    # iterm -> (3,6,12,24), other
    data["iterm"] = data["iterm"].apply(lambda x: 0 if x in [3, 6, 12, 24] else 1)
    # hcefg -> (NaN, 1, 8), (0, 3, 9), 6, others
    data["hcefg"] = data["hcefg"].fillna(1)
    data["hcefg"] = data["hcefg"].apply(
        lambda x: 0 if x in [0, 3, 9] else 1 if x in [1, 8] else 2 if x == 6 else 3
    )

    # sparse cate: [mcc, stocn, scity, csmcu]
    preprocess_sparsecate(data)

    # sort by cano and abs_time
    data = data.sort_values(by=["cano", "abs_time"])
    return data


def preprocess_history(data):
    """
    Preprocesses the historical data by splitting it into train, validation, and test sets based on time ranges.
    For each time range, it performs additional processing steps such as handling records with odd number of occurrences,
    calculating fraud rates, and saving the processed data.

    Args:
        data (DataFrame): The historical data to be preprocessed.

    Returns:
        None
    """
    # split train, val
    # train: >=0, <50
    # val: >=50, <56
    # public: >=56, <60
    # private1: >=60, <65
    data = data.sort_values(by=["cano", "abs_time"]).reset_index(drop=True)

    # group by cano, and sort by abs_time, then diff abs_time
    data["time_diff"] = data.groupby("cano")["abs_time"].diff()
    data["time_diff"] = data["time_diff"].fillna(-1)

    # train
    for time_range in [50, 56, 60]:
        train = data[data["abs_time"] < time_range]
        # find those cano has odd number of records
        odd = train.groupby("cano").filter(lambda x: len(x) % 2 != 0)
        # get the first record from cano that has odd number of records
        first_of_odd = odd.groupby("cano").head(1)
        # the rest of cano will only have even number of records
        train_multi = train.drop(first_of_odd.index)
        # add more to train_1
        first_label_1 = train_multi[train_multi["label"] == 1].groupby("cano").head(1)
        first_label_0 = train_multi[train_multi["label"] == 0].groupby("cano").head(1)
        train_1 = pd.concat(
            [first_of_odd, first_label_1, first_label_0], axis=0
        ).sort_index()
        # process train_multi
        cano_fraudrate = train.groupby("cano")["label"].mean()
        train_multi["fraudrate"] = train_multi["cano"].apply(
            lambda x: cano_fraudrate[x]
        )
        train_multi["prev_label"] = train_multi.groupby("cano")["label"].shift(1)
        train_multi = train_multi[~train_multi["prev_label"].isna()]
        train_multi["prev_label"] = train_multi["prev_label"].astype(int)
        # save
        train_1 = train_1.drop(columns=["txkey", "cano", "abs_time", "time_diff"])
        train_multi = train_multi.drop(columns=["txkey", "cano", "abs_time"])
        save_data(
            f"data/history/train{time_range}",
            {"train_1": train_1, "train_multi": train_multi},
        )

    # val and test
    for name, (k1, k2) in {
        "val": (50, 56),
        "public": (56, 60),
        "private1": (60, 65),
    }.items():
        history = data[data["abs_time"] < k2]
        cur = history[history["abs_time"] >= k1]
        past = history[history["abs_time"] < k1]

        cano_fraudrate = past.groupby("cano")["label"].mean()
        cur_canos = set(cur["cano"].unique())
        past_canos = set(past["cano"].unique())

        # canos that have history records
        have_history = cur[cur["cano"].isin(past_canos)]
        last_record = past[past["cano"].isin(cur_canos)].groupby("cano").tail(1)
        have_history = pd.concat([have_history, last_record], axis=0).sort_values(
            by=["cano", "abs_time"]
        )
        have_history["fraudrate"] = have_history["cano"].apply(
            lambda x: cano_fraudrate[x]
        )
        have_history["prev_label"] = have_history.groupby("cano")["label"].shift(1)
        have_history = have_history[~have_history["prev_label"].isna()]
        have_history["prev_label"] = have_history["prev_label"].astype(int)
        # canos that have no history records
        no_history = cur[cur["cano"].isin(cur_canos - past_canos)]
        no_history_first = no_history.groupby("cano").head(1)
        no_history_others = no_history.drop(
            no_history_first.index
        )  # the rest of cano that has no history
        no_history_others["fraudrate"] = 0
        # save
        have_history = have_history.drop(columns=["abs_time"])
        no_history_others = no_history_others.drop(columns=["abs_time"])
        no_history_first = no_history_first.drop(columns=["abs_time", "time_diff"])
        save_data(
            f"data/history/{name}",
            {
                f"have_history": have_history,
                f"no_history_first": no_history_first,
                f"no_history_others": no_history_others,
            },
        )


def preprocess_sparsecate(data):
    """
    Preprocesses the sparse categorical columns in the given data.

    Args:
        data (pandas.DataFrame): The input data containing sparse categorical columns.

    Returns:
        pandas.DataFrame: The preprocessed data with sparse categorical columns transformed.

    """
    # sparse cate
    # cateA=>0, cateB=>1, cateC=>2
    # sparse_cate_cols = ["mcc", "stocn", "scity", "csmcu"]
    # mcc
    data["mcc"] = data["mcc"].fillna(-1).astype(int)
    cateA = {493, 494, 289, 375, 288, 276, 273, 215, 272, 499, 320, 322, 275, 282, 309}
    cateC = {324}
    data["mcc"] = data["mcc"].apply(
        lambda x: 0 if x in cateA else 2 if x in cateC else 1
    )

    # stocn
    data["stocn"] = data["stocn"].fillna(-1).astype(int)
    cateA = {0}
    cateB = {112, 25, 113, 42}
    data["stocn"] = data["stocn"].apply(
        lambda x: 0 if x in cateA else 1 if x in cateB else 2
    )

    # scity
    data["scity"] = data["scity"].fillna(-1).astype(int)
    cateA = {15742, 6464, 15808, 5623, 15760, 11380, 16117, 15759, 6469, 16115, 13451}
    cateC = {11170, 10300}
    data["scity"] = data["scity"].apply(
        lambda x: 0 if x in cateA else 2 if x in cateC else 1
    )

    # csmcu -> (70, NaN), (68, 81), others
    data["csmcu"] = data["csmcu"].fillna(70).astype(int)
    cateB = {68, 81}
    data["csmcu"] = data["csmcu"].apply(
        lambda x: 0 if x == 70 else 1 if x in cateB else 2
    )

    return data


def get_onehot(data, cols):
    """
    Apply one-hot encoding to the specified columns in the given DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        cols (list): A list of column names to apply one-hot encoding.

    Returns:
        pandas.DataFrame: The DataFrame with one-hot encoded columns.
    """
    for col in cols:
        data[col] = data[col].astype(int)
        data = pd.concat(
            [data, pd.get_dummies(data[col], prefix=col, dtype=int)], axis=1
        )
    data.drop(columns=cols, inplace=True)
    return data


if __name__ == "__main__":
    # load data
    trainval, public, private1 = load_data(
        "data/dataset", ["training", "public", "private_1_processed"]
    )
    # phase 1
    trainval = preprocess_base(trainval, is_train=True)
    public = preprocess_base(public, is_train=False)
    private1 = preprocess_base(private1, is_train=False)
    # save
    save_data(
        "data/base", {"trainval": trainval, "public": public, "private1": private1}
    )

    # load data
    trainval, public, private1 = load_data(
        "data/base", ["trainval", "public", "private1"]
    )
    # phase 2
    trainval = preprocess_xgb(trainval)
    public = preprocess_xgb(public)
    private1 = preprocess_xgb(private1)
    private1["label"] = -1  # to align with public and trainval
    # one-hot
    # iterm is already bool
    cate_cols = ["contp", "etymd", "hcefg", "mcc", "stocn", "scity", "csmcu"]
    data = get_onehot(pd.concat([trainval, public, private1], axis=0), cate_cols)
    # split data back
    trainval = data.iloc[: len(trainval)]
    public = data.iloc[len(trainval) : len(trainval) + len(public)]
    private1 = data.iloc[len(trainval) + len(public) :]
    # save
    save_data(
        "data/xgb", {"trainval": trainval, "public": public, "private1": private1}
    )

    # load data
    trainval, public, private1 = load_data(
        "data/xgb", ["trainval", "public", "private1"]
    )
    # phase 3
    data = pd.concat([trainval, public, private1], axis=0)
    preprocess_history(data)

    exit()
