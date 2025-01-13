#%%
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


#%% 설정
repeat_num = 1000
n_list = [50, 100, 500,]

for n in n_list:
    model1_coef_list = []
    model2_coef_list = []
    for repeat_seed in tqdm(range(1, repeat_num+1)):
        np.random.seed(repeat_seed)

        # 1. 독립 변수 X 생성
        x1 = np.random.normal(0, 1, (n, 1))  # Normal(0,1)
        x2 = np.random.normal(10, 1, (n, 1))  # Normal(0,1)

        y1 = np.random.normal(3*x1, 1)  # Normal(0,1)
        y2 = np.random.normal(3*x2, 1)  # Normal(0,1)

        ##### randomized data fitting #####
        model1 = LinearRegression(fit_intercept=True)
        model1.fit(x1, y1)

        model2 = LinearRegression(fit_intercept=True)
        model2.fit(x2, y2)

        model1_coef_list.append(np.concatenate([model1.coef_[0], model1.intercept_]))
        model2_coef_list.append(np.concatenate([model2.coef_[0], model2.intercept_]))

    model1_coef_arr = np.array(model1_coef_list)
    model2_coef_arr = np.array(model2_coef_list)

    print(f"# of samples: {n}")
    print(f"[y|x1] coef: {model1_coef_arr[:,0].mean()} / bias : {model1_coef_arr[:,1].mean()}")
    print(f"[y|x2] coef: {model2_coef_arr[:,0].mean()} / bias : {model2_coef_arr[:,1].mean()}")
    print()
# %%
