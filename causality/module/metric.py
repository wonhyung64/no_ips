import numpy as np
from collections import defaultdict


def ncdcg_func(pred, x_test, y_test, top_k_list):
    """
    Evaluate cDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            true_top_k_rel = true_u[np.argsort(-true_u)][:top_k]
            cdcg_k = pred_top_k_rel / log2_iplus1
            icdcg_k = true_top_k_rel / log2_iplus1
            ncdcg_k = np.sum(cdcg_k) / (np.sum(icdcg_k)+1e-9)
            result_map[f"ncdcg_{top_k}"].append(ncdcg_k)

    return result_map


def cdcg_func(pred, x_test, y_test, top_k_list):
    """
    Evaluate cDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            cdcg_k = pred_top_k_rel / log2_iplus1
            result_map[f"cdcg_{top_k}"].append(cdcg_k)

    return result_map


def cp_func(pred, x_test, y_test, top_k_list):
    """
    Evaluate cp@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            cp_k = sum(pred_top_k_rel)

            result_map[f"cp_{top_k}"].append(cp_k)

    return result_map


def car_func(pred, x_test, y_test, top_k_list):
    """
    Evaluate cp@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            car_k = np.sum(-(np.arange(1,top_k+1)) * pred_top_k_rel)

            result_map[f"car_{top_k}"].append(car_k)

    return result_map
