import pandas as pd
import numpy as np
from utils import load_data
import sys


def check_null(data):
    col_names = data.columns
    size = len(data)
    for col in col_names:
        has_null = data[col].isnull().sum() > 0
        if has_null:
            print(f"{col} \t {data[col].isnull().sum() / size * 100:.2f} %")

    print("=====================================")


def compute_corr(train):
    # check corr between attributes and label
    bool_cols = ["ecfg", "insfg", "bnsfg", "ovrlt", "flbmk", "flg_3dsmk"]
    for col in bool_cols:
        try:
            corr = train["label"].corr(train[col])
            print(f"{col}\t{corr:.5f}")
        except:
            continue


def analyze_col(col, train, test, k=7):
    print("=====================================")
    # build a df
    # col 1: col name => top k popular index in train data
    # col 2: train => portion of top k popular index  in train data
    # col 3: normal rate => portion of label==0 of top k popular index in train data

    # train
    train_vc = train[col].value_counts(normalize=True, dropna=False)
    df = pd.DataFrame()
    df[col] = train_vc.index[:k]
    df.set_index(col, inplace=True)
    df["portion"] = train_vc.values[:k]
    normal_rates = []
    mask_topk = np.zeros(len(train), dtype=bool)
    for idx in range(len(df)):
        if np.isnan(df.index[idx]):
            mask = train[col].isna()
        else:
            mask = train[col] == df.index[idx]
        normal_rates.append(
            train[mask]["label"].value_counts(normalize=True, dropna=False)[0]
        )
        mask_topk |= mask
    df["normal_rate"] = normal_rates
    if mask_topk.sum() < len(train):
        # add a row: 'others', portion, normal_rate
        df.loc["others"] = [
            train_vc.values[k:].sum(),
            train[~mask_topk]["label"].value_counts(normalize=True, dropna=False)[0],
        ]
    print(df.round(4))

    # test
    if test is not None:
        test_vc = test[col].value_counts(normalize=True, dropna=False)
        df = pd.DataFrame()
        df[col] = test_vc.index[:k]
        df.set_index(col, inplace=True)
        df["portion"] = test_vc.values[:k]
        print(df.round(4))

    if len(train_vc) > 2:
        print(
            "train: (min, max, mean, media)",
            f"{train[col].min():.2f}",
            f"{train[col].max():.2f}",
            f"{train[col].mean():.2f}",
            f"{train[col].median():.2f}",
        )
        print(
            "test: (min, max)", f"{test[col].min():.2f}", f"{test[col].max():.2f}"
        ) if test is not None else None


def find_significant_cate(col, data):
    # criteria: 1) proportion > ?% 2) p(label=0) > 0.99?
    df = data[col].value_counts(normalize=True, dropna=False)
    k = 20
    proportion_th = df.iloc[k] if df.iloc[k] > 0.001 else 0.001
    print(f"proportion_th: {proportion_th:.4f}", f"len(df): {len(df)}")
    df = df[df > proportion_th]
    df = pd.DataFrame(df)

    mask_trusted = np.zeros(len(data), dtype=bool)
    for cate in df.index:
        if np.isnan(cate):
            mask = data[col].isna()
        else:
            mask = data[col] == cate
        mask_trusted |= mask
        normal_rate = data[mask]["label"].value_counts(normalize=True, dropna=False)[0]
        df.loc[cate, "normal_rate"] = normal_rate

    if mask_trusted.sum() < len(data):
        # add a row: 'others', portion, normal_rate
        df.loc["others"] = [
            1 - df["proportion"].sum(),
            data[~mask_trusted]["label"].value_counts(normalize=True, dropna=False)[0],
        ]

    # sort by normal_rate
    df.sort_values(by="normal_rate", ascending=False, inplace=True)
    print(df.round(4), "\n")


def compute_effective_digits(col, data):
    dd = data[col]
    target_nunique = dd.nunique()
    for k in range(4, len(str(dd[0]))):
        tmp = dd.apply(lambda x: x[-k:])
        if tmp.nunique() == target_nunique:
            print(f"{col} => {k} digits")
            break
    assert dd.nunique() == dd.apply(lambda x: x[-k:]).nunique()


def analyze_cano(data):
    data = data[["locdt", "loctm", "cano", "label"]]
    # group training data by cano
    data = data.groupby("cano")

    # how many unique cano
    print(f"unique cano: {len(data)}")
    # how many cano has only one record
    print(f"cano has only one record: {len(data.filter(lambda x: len(x)==1))}")

    # filter out those cano has any record with label==1
    exp_data = data.filter(lambda x: x["label"].sum() > 0)
    exp_data = exp_data.groupby("cano")

    # how many cano has any record with label==1
    print(f"cano has any record with label==1: {len(exp_data)}")

    # how many cano has only one record ant the record is label==1
    print(
        f"cano has only one record and the record is label==1: {len(exp_data.filter(lambda x: len(x)==1))}"
    )

    # each cano has how many records
    sizes = exp_data.apply(lambda x: len(x))
    # ratio of label==1 in each cano
    stat = exp_data["label"].sum() / sizes
    # how many stat == 1.0
    print(f"stat == 1.0: {sum(stat == 1.0)}")
    print(f"stat > 0.5: {sum(stat > 0.5)}")

    # sort by locdt and loctm
    exp_data = exp_data.apply(
        lambda x: x.sort_values(by=["locdt", "loctm"])
    ).reset_index(drop=True)
    #  count label transition rate by cano
    label_transition_rate = exp_data.groupby("cano")["label"].apply(
        lambda x: x.diff().fillna(0).abs().sum() / (len(x))
    )
    print(
        f"label_transition_rate: {label_transition_rate.mean():.4f}+-{label_transition_rate.std():.4f}"
    )
    print(
        f"label_transition_rate median, min, max: {label_transition_rate.median():.4f}, {label_transition_rate.min():.4f}, {label_transition_rate.max():.4f}"
    )


if __name__ == "__main__":
    hash_cols = ["txkey", "chid", "cano", "mchno", "acqic"]

    time_cols = ["locdt", "loctm"]
    bool_cols = ["ecfg", "insfg", "bnsfg", "ovrlt", "flbmk", "flg_3dsmk"]
    num_cols = ["conam", "flam1", "csmam"]
    cate_cols = ["contp", "etymd", "iterm", "stscd", "hcefg"]
    sparse_cate_cols = ["mcc", "stocn", "scity", "csmcu"]

    train, test = load_data("data/dataset", ["training", "public_processed"])

    print("training null rate: ")
    check_null(train)
    print("testing null rate: ")
    check_null(test)

    print("corr of cate cols and the label:")
    compute_corr(train)

    org_stdout = sys.stdout
    sys.stdout = open("analysis/analysis.txt", "w")
    cols = time_cols + bool_cols + num_cols + cate_cols + sparse_cate_cols
    for col in cols:
        analyze_col(col, train, test)
    sys.stdout = org_stdout

    sys.stdout = open("analysis/significant_cate.txt", "w")
    for col in sparse_cate_cols:
        find_significant_cate(col, train)
    sys.stdout = org_stdout

    sys.stdout = open("analysis/cano.txt", "w")
    analyze_cano(train)
    sys.stdout = org_stdout

    exit()
