#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


#%%
data_path = "/Users/wonhyung64/Github/no_ips/causality/data/sensitivity.csv"
df = pd.read_csv(data_path)
df.iloc[:, 5:] = df.iloc[:, 5:] * 100
performance = df.groupby(["dataset_name", "loss_type", "alpha"]).mean(["accuracy_y1", "accuracy_y0", "auc_y1", "auc_y0"]).reset_index()



#%%
dataset_name_list = ["original", "personalized"]
outcome_list = ["y1", "y0"]

loss_type_list = ["ips", "naive"]
alpha_list = [0., 0.0001, 0.001, 0.01, 0.1, 1., 10.]
alpha_ticks = [-5, -4, -3, -2, -1, 0, 1]
alpha_labels = [r"", r"$0$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]

font_size=16
# font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
# colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# fontprop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = fontprop.get_name()
# plt.rcParams['font.weight'] = 'bold'


#%%
cond_ips = performance["loss_type"] == "ips"
cond_naive = performance["loss_type"] == "naive"
for dataset_name in dataset_name_list:
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
    for i, outcome in enumerate(outcome_list):
        cond_data = performance["dataset_name"] == dataset_name

        axes[i].grid(True, linestyle='--', alpha=0.5)
        auc_ips = performance[cond_data & cond_ips].sort_values(by="alpha")[f"auc_{outcome}"].to_numpy()
        auc_naive = performance[cond_data & cond_naive].sort_values(by="alpha")[f"auc_{outcome}"].to_numpy()

        acc_ips = performance[cond_data & cond_ips].sort_values(by="alpha")[f"accuracy_{outcome}"].to_numpy()
        acc_naive = performance[cond_data & cond_naive].sort_values(by="alpha")[f"accuracy_{outcome}"].to_numpy()
        if i == 0:
            axes[i].plot(alpha_ticks, auc_ips, label="IPS", c="#d62728", marker="s")
            axes[i].plot(alpha_ticks, auc_naive, label="Naive", c="#1f77b4", marker="o")
        else:
            axes[i].plot(alpha_ticks, auc_ips, c="#d62728", marker="s")
            axes[i].plot(alpha_ticks, auc_naive, c="#1f77b4", marker="o")

        axes[i].set_xticklabels(alpha_labels)
        axes[i].set_xlabel(r"$\lambda$")
        axes[i].set_ylabel(r"$AUC$")
        

        # axes[1].plot(alpha_ticks, acc_ips)
        # axes[1].plot(alpha_ticks, acc_naive)
        # axes[1].set_xticklabels(alpha_labels)
        # axes[1].set_xlabel(r"$\lambda$")
        # axes[1].set_ylabel("Acc")

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, edgecolor="black")
    axes[0].set_title(r"$\Pr(Y=1|T=1,x)$")
    axes[1].set_title(r"$\Pr(Y=1|T=0,x)$")
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0., 0.])
    plt.show()
    plt.close()

    img_name = f"{dataset_name}"
    print(img_name)
    fig.savefig(f"./{img_name}.pdf", bbox_inches='tight')



# %%
