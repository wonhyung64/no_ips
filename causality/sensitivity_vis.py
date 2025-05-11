#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


#%%
# data_path = "/Users/wonhyung64/Github/no_ips/causality/data/sensitivity.csv"
data_path = "/Users/wonhyung64/Desktop/true_propensity.csv"
df = pd.read_csv(data_path)
performance = df.groupby(["dataset", "loss", "alpha"]).mean(["cdcg_10", "cdcg_1372", "cp_10", "cp_100", "auc_y1", "auc_y0"]).reset_index()


#%%
dataset_name_list = ["original", "personalized"]
outcome_list = ["y1", "y0", "1372"]

loss_type_list = ["ips", "naive"]
alpha_list = [0., 0.0001, 0.001, 0.01, 0.1, 1., 10.]
alpha_ticks = [-5, -4, -3, -2, -1, 0, 1]
alpha_labels = [r"", r"$0$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]

font_size=14


#%%
cond_ips = performance["loss"] == "ips"
cond_naive = performance["loss"] == "naive"
for dataset_name in dataset_name_list:
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(5,5))

    cond_data = performance["dataset"] == dataset_name

    y1_ips = performance[cond_data & cond_ips].sort_values(by="alpha")[f"auc_y1"].to_numpy()
    y1_naive = performance[cond_data & cond_naive].sort_values(by="alpha")[f"auc_y1"].to_numpy()

    y0_ips = performance[cond_data & cond_ips].sort_values(by="alpha")[f"auc_y0"].to_numpy()
    y0_naive = performance[cond_data & cond_naive].sort_values(by="alpha")[f"auc_y0"].to_numpy()

    cate_ips = performance[cond_data & cond_ips].sort_values(by="alpha")[f"cdcg_1372"].to_numpy()
    cate_naive = performance[cond_data & cond_naive].sort_values(by="alpha")[f"cdcg_1372"].to_numpy()

    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].plot(alpha_ticks, y1_ips, label="IPS", c="#1f77b4", marker="s")
    axes[0].plot(alpha_ticks, y1_naive, label="Naive", c="#d62728", marker="o")
    axes[0].set_xticklabels(alpha_labels, fontsize=font_size)
    axes[0].tick_params(axis="y", labelsize=font_size)
    axes[0].set_xlabel(r"$\lambda$", fontsize=font_size)
    axes[0].set_ylabel(r"$AUC$", fontsize=font_size)
    axes[0].set_title(r"$\Pr(Y=1|T=1,x)$", fontsize=font_size)

    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].plot(alpha_ticks, y0_ips, c="#1f77b4", marker="s")
    axes[1].plot(alpha_ticks, y0_naive, c="#d62728", marker="o")
    axes[1].set_xticklabels(alpha_labels, fontsize=font_size)
    axes[1].tick_params(axis="y", labelsize=font_size)
    axes[1].set_xlabel(r"$\lambda$", fontsize=font_size)
    axes[1].set_ylabel(r"$AUC$", fontsize=font_size)
    axes[1].set_title(r"$\Pr(Y=1|T=0,x)$", fontsize=font_size)


    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].plot(alpha_ticks, cate_ips, c="#1f77b4", marker="s")
    axes[2].plot(alpha_ticks, cate_naive, c="#d62728", marker="o")
    axes[2].set_xticklabels(alpha_labels, fontsize=font_size)
    axes[2].tick_params(axis="y", labelsize=font_size)
    axes[2].set_xlabel(r"$\lambda$", fontsize=font_size)
    axes[2].set_ylabel(r"$cDCG$", fontsize=font_size)
    axes[2].set_title(r"$\Pr(Y=1|T=0,x)-\Pr(Y=1|T=0,x)$", fontsize=font_size)


    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=font_size)
    plt.tight_layout()
    plt.show()
    plt.close()

    img_name = f"{dataset_name}"
    print(img_name)
    fig.savefig(f"./{img_name}.pdf", bbox_inches='tight')



# %%
