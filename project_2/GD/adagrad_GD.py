import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(1)

n = 10000
x = np.linspace(-1,1, n).reshape(-1,1)
y = 2 + 3 * x + 4 * np.multiply(x,x) + 0.3 * np.random.randn(n,1)

X = np.ones((n, 3))

for i in range(n):
    X[i, 1] = x[i]
    X[i, 2] = x[i] * x[i]


x_train, x_test, y_train, y_test, X_train, X_test = train_test_split(x,y,X)

eps = pow(10,-8)
delta = pow(10,-8)

#gammas_0 = np.arange(0.1, 0.95, 0.05)
gamma_0 = 0.2

lmbs = np.arange(0.001, 10, 0.5)
#lmb = 0.01


I = np.zeros((3,3))
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1


G = np.zeros((3,3))

#n_iter = np.zeros((len(gammas_0), 1))
#for j in range(len(gammas_0)):
n_iter = np.zeros((len(lmbs), 1))
for j in range(len(lmbs)):
    beta = np.random.randn(3,1)
    beta_current_iter = np.random.randn(3,1)

    #gamma_0 = gammas_0[j]
    lmb = lmbs[j]
    i = 0
    while mean_squared_error(beta, beta_current_iter)>eps:
        #gradient = 2/n * X_train.T @ (X_train @ beta - y_train)
        gradient = 2/n * X_train.T @ (X_train @ beta - y_train) + lmb * I @ beta
        beta_current_iter = beta

        G = G + (gradient @ gradient.T)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = gamma_0/(delta + sqrt_G).reshape(-1,1)

        beta = beta - gamma * gradient
        i = i+1


    beta_ols = np.linalg.pinv(X_test.T @ X_test) @ X_test.T @ y_test

    y_ols = X_test @ beta_ols
    y_grad = X_test @ beta

    mse_grad = mean_squared_error(y_test, y_grad)
    mse_ols = mean_squared_error(y_test, y_ols)

    print(i)
    #print(gamma)
    #print(lmb)
    n_iter[j] = i



plt.figure(1)
#plt.plot(gammas_0, n_iter)
#plt.xlabel('Parameter $\gamma_0$')
plt.ylabel('Numer of iterations $n_{iter}$')
plt.plot(lmbs, n_iter)
plt.xlabel('Penalty parameter $\lambda$')
plt.show()