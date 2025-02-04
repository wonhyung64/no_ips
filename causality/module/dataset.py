import os
import numpy as np
import pandas as pd


def load_data(data_dir, dataset_name):
    if dataset_name == "original":
        sub_dir = f"dunn_cat_mailer_10_10_1_1/original_rp0.40"
    elif dataset_name == "personalized":
        sub_dir = f"dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210"

    dataset_dir = os.path.join(data_dir, sub_dir)

    train_df = pd.read_csv(f"{dataset_dir}/data_vali.csv")
    test_df = pd.read_csv(f"{dataset_dir}/data_test.csv")

    x_train = train_df[["idx_user", "idx_item", "outcome", "treated", "propensity"]].to_numpy()
    x_test = test_df.groupby(["idx_user", "idx_item"]).mean("causal_effect")["causal_effect"].reset_index().to_numpy()

    print(f"Loaded from {dataset_name} dataset")
    print("[train] num data:", x_train.shape[0])
    print("[test]  num data:", x_test.shape[0])
    print("user, item indices start from '0'.")

    return x_train, x_test


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)
