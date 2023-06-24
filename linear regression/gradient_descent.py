import numpy as np
import pandas as pd

# データ準備
df = df = pd.read_csv('data.csv')
x = df['space'].values
y = df['rent'].values

# 損失関数を定義
def loss_func(theta_0, theta_1, x, y):
    return np.mean((y-(theta_0 + theta_1 * x))**2)


# print(loss_func(2, 2, x, y))

# θ0とθ1の初期値を設定
theta_0_init = -5
theta_1_init = 5

# イテレーションの数
epoch = 10**5

# 学習率
alpha = 0.0001

theta_0_hist = []
theta_1_hist = []

# 初期化
theta_0_hist.append(theta_0_init)
theta_1_hist.append(theta_1_init)


# θ0とθ1を更新する関数
def update_theta0(theta_0, theta_1, x, y, alpha=0.001):
    return theta_0 - alpha * 2 * np.mean((theta_0 + theta_1 * x) - y)


# update_theta0(theta_0_init, theta_1_init, x, y)

def update_theta1(theta_0, theta_1, x, y, alpha=0.001):
    return theta_1 - alpha * 2 * np.mean(((theta_0 + theta_1 * x) - y) * x)


# update_theta1(theta_0_init, theta_1_init, x, y)

# θ0とθ1をイレーションにより更新
for _ in range(epoch):
    update_theta_0 = update_theta0(theta_0_hist[-1], theta_1_hist[-1], x=x, y=y, alpha=alpha)
    update_theta_1 = update_theta1(theta_0_hist[-1], theta_1_hist[-1], x=x, y=y, alpha=alpha)
    theta_0_hist.append(update_theta_0)
    theta_1_hist.append(update_theta_1)

# 損失関数の結果の推移
rent_cost = [loss_func(*param, x=x, y=y) for param in zip(theta_0_hist, theta_1_hist)]
print(rent_cost[-1])

# 回帰モデル使ってspaceから　rentを予測
space = 20
rent = theta_0_hist[-1] + theta_1_hist[-1]*space
print(rent)




