import pandas as pd
from data import GT, LT, DSThreshold, SupportedDatasets

def load_single_target_tba():
    df = pd.read_excel("./datasets/GSK_TBA_AH_JSFedit070425.xlsx")
    df = df.loc[:, ["smiles", "parent_remaining_24", "metabolite_detected"]]

    df["cont_target"] = df["parent_remaining_24"].round(1) / 100
    df["bin_target"] = df["cont_target"] > 0.5
    return df, GT(0.50)


def load_dual_target_tba():
    df = pd.read_excel("./datasets/GSK_TBA_AH_JSFedit070425.xlsx")
    df = df.loc[:, ["smiles", "parent_remaining_24", "metabolite_detected"]]

    df["cont_target"] = (
        df["parent_remaining_24"].round(1) / 100 + df["metabolite_detected"]
    )
    df["bin_target"] = df["cont_target"] > 1.5
    return df, GT(1.5)


def load_gsk_hepg2():
    df = pd.read_csv("./datasets/GSK_HepG2.csv")
    df = df.loc[:, ["SMILES", "% inhibition of HepG2 cell line: PCT_INHIB_HEPG2 (%)"]]
    df.columns = ["smiles", "per_inhibition"]

    df["cont_target"] = df["per_inhibition"] / 100
    df["bin_target"] = df["cont_target"] > 0.5
    return df, GT(0.5)


def load_derbyshire_malaria():
    df = pd.read_csv("./datasets/Derbyshire_malaria.csv")
    cols = ["Compound SMILES", "Parasite % Control Avg"]
    df = df.loc[:, cols]
    df.columns = ["smiles", "parasite_remain_per"]

    df["cont_target"] = df["parasite_remain_per"] / 100
    df["bin_target"] = df["cont_target"] < 0.15
    return df, LT(0.15)


def load_derbyshire_hepg2():
    df = pd.read_csv("./datasets/Derbyshire_malaria.csv")
    cols = ["Compound SMILES", "Liver % Control Avg"]
    df = df.loc[:, cols]
    df.columns = ["smiles", "liver_remain_per"]

    df["cont_target"] = df["liver_remain_per"] / 100
    df["bin_target"] = df["cont_target"] < 0.5
    return df, LT(0.5)


def load_pk():
    df = pd.read_csv("./datasets/PK.csv")
    df = df.loc[:, ["mol", "AUC"]]
    df.columns = ["smiles", "auc"]
    df = df.dropna().reset_index(drop=True)

    df["cont_target"] = df["auc"]
    df["bin_target"] = df["cont_target"] > 1000
    return df, GT(1000)


def load_dataset(dataset: SupportedDatasets) -> tuple[pd.DataFrame, DSThreshold]:
    if dataset == SupportedDatasets.SINGLE_TARGET_TBA:
        df, df_classification_threshold = load_single_target_tba()
    elif dataset == SupportedDatasets.DUAL_TARGET_TBA:
        df, df_classification_threshold = load_dual_target_tba()
    elif dataset == SupportedDatasets.GSK_HEPG2:
        df, df_classification_threshold = load_gsk_hepg2()
    elif dataset == SupportedDatasets.PK:
        df, df_classification_threshold = load_pk()
    elif dataset == SupportedDatasets.DB_MALARIA:
        df, df_classification_threshold = load_derbyshire_malaria()
    elif dataset == SupportedDatasets.DB_HEPG2:
        df, df_classification_threshold = load_derbyshire_hepg2()
    else:
        raise ValueError(dataset)
    
    return df, df_classification_threshold