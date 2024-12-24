import os
import numpy as np
import pandas as pd


def load_data(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir, dataset_name)

    if dataset_name == "yahoo_r3":
        train_file = os.path.join(dataset_dir, "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(dataset_dir, "ydata-ymusic-rating-study-v1_0-test.txt")
        x_train = []
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)
        x_test = []
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)

    elif dataset_name == "coat":
        train_file = os.path.join(dataset_dir, "train.csv")
        test_file = os.path.join(dataset_dir, "test.csv")

        x_train = pd.read_csv(train_file).to_numpy()
        x_train = np.stack([x_train[:,0]+1, x_train[:,1]+1, x_train[:,2]], axis=-1)

        x_test = pd.read_csv(test_file).to_numpy()
        x_test = np.stack([x_test[:,0]+1, x_test[:,1]+1, x_test[:,2]], axis=-1)

    print(f"Loaded from {dataset_name} dataset")
    print("[train] num data:", x_train.shape[0])
    print("[test]  num data:", x_test.shape[0])
    print("user, item indices start from '1'.")

    return x_train, x_test


def binarize(y, thres=3):
    y[y< thres] = 0
    y[y>=thres] = 1

    return y
