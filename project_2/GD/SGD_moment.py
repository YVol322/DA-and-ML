import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(1)

n = 10000 #number of points
x = np.linspace(-1,1, n).reshape(-1,1)
y = 2 + 3 * x + 4 * np.multiply(x,x) + 0.3 * np.random.randn(n,1)

X = np.ones((n, 3))

for i in range(n):
    X[i, 1] = x[i]
    X[i, 2] = x[i] * x[i]

x_train, x_test, y_train, y_test, X_train, X_test = train_test_split(x,y,X)

eps = 1e-8
delta = 0.7

t0, t1 = 15, 20

def learning_schedule(t):
    return t0/(t+t1)

M = 200 #size of each mini bach
m = int(n/M) #number of minibatches

n_epochs_list = np.arange(1,20,1) #number of epochs

mses = np.zeros((len(n_epochs_list), 1))
mses_OLS = np.zeros((len(n_epochs_list), 1))
mses_R = np.zeros((len(n_epochs_list), 1))

lmb = 0.01

I = np.zeros((3,3))
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1

for j in range(len(n_epochs_list)):
    n_epochs = n_epochs_list[j]

    beta = np.random.randn(3,1)
    betas = []
    betas.append(np.zeros((3,1)))
    betas.append(beta)


    for epoch in range(n_epochs):
        for i in range(m):
            k = np.random.randint(m)
            X_k = X_train[k*M:(k+1)*M,:]
            y_k = y_train[k*M:(k+1)*M,:]
            #gradients = (2.0/M)* X_k.T @ ((X_k @ betas[-1])-y_k)
            gradients = (2.0/M)* X_k.T @ ((X_k @ betas[-1])-y_k) + lmb * I @ betas[-1]
            gamma = learning_schedule(epoch*m+i)
            if i == 0:
                betas.append(betas[i+1] - gamma * gradients)
            else:
                betas.append(betas[i+1] - gamma * gradients + delta*(betas[-1]-betas[-2]))




    #beta_ols = np.linalg.pinv(X_test.T @ X_test) @ X_test.T @ y_test
    #y_ols = X_test @ beta_ols

    beta_R = np.linalg.pinv(X_test.T @ X_test + lmb * I) @ X_test.T @ y_test
    y_R = X_test @ beta_R

    y_grad = X_test @ betas[-1]

    mse_grad = mean_squared_error(y_test, y_grad)
    #mse_ols = mean_squared_error(y_test, y_ols)
    mse_R = mean_squared_error(y_test, y_R)

    mses[j]= mean_squared_error(y_test, y_grad)
    #mses_OLS[j] = mean_squared_error(y_test, y_ols)
    mses_R[j] = mean_squared_error(y_test, y_R)


plt.figure(1)
plt.plot(n_epochs_list, mses, label = 'SGD MSEs')
#plt.plot(n_epochs_list, mses_OLS, label = 'OLS MSE')
plt.plot(n_epochs_list, mses_R, label = 'Rigde MSE')
plt.xlabel('Number of epochs $n_{epochs}$')
plt.ylabel('MSE')
plt.legend()
plt.show()