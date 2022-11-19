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

gammas = np.arange(0.05, 0.95, 0.05)
#gamma = 0.8

delta = 0.5
Niterations = pow(10,6)
eps = pow(10,-8)

lmb = 0.5
#lmbs = np.arange(0.001, 10, 0.05)

I = np.zeros((3,3))
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1

n_iter = np.zeros((len(gammas), 1))
for j in range(len(gammas)):
#n_iter = np.zeros((len(lmbs), 1))
#for j in range(len(lmbs)):
    beta = np.random.randn(3,1)
    betas = []
    betas.append(np.zeros((3,1)))
    betas.append(beta)

    gamma = gammas[j]
    #lmb = lmbs[j]

    i=0
    while mean_squared_error(betas[i+1], betas[i])>eps:
        gradient = 2/n * X_train.T @ (X_train @ betas[i+1] - y_train)
        #gradient = 2/n * X_train.T @ (X_train @ betas[i+1] - y_train) + lmb * I @ beta
        if i == 0:
            betas.append(betas[i+1] - gamma * gradient)
            i = i+1
        else:
            betas.append(betas[i+1] - gamma * gradient + delta*(betas[i+1]-betas[i]))
            i = i+1
        n_iter[j] = i




    beta_ols = np.linalg.inv(X_test.T @ X_test) @ X_test.T @ y_test


    y_ols = X_test @ beta_ols
    y_grad = X_test @ betas[-1]

    mse_grad = mean_squared_error(y_test, y_grad)
    mse_ols = mean_squared_error(y_test, y_ols)

    print(i)
    print(gamma)

plt.figure(1)
plt.plot(gammas, n_iter)
plt.xlabel('Learning rate $\gamma$')
plt.ylabel('Numer of iterations $n_{iter}$')
#plt.plot(lmbs, n_iter)
#plt.xlabel('Penalty parameter $\lambda$')
plt.show()