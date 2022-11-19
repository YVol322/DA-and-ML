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

beta = np.random.randn(3,1)
beta_current = np.random.randn(3,1)

eps = 1e-8

t0, t1 = 15, 20

def learning_schedule(t):
    return t0/(t+t1)

M = 2500 #size of each mini bach
m = int(n/M) #number of minibatches

#n_epochs_list = np.arange(1,20,1)
#mses = np.zeros((len(n_epochs_list), 1))
#mses_OLS = np.zeros((len(n_epochs_list), 1))
#lmb = 0.01
lmbs = np.arange(0, 0.5, 0.05)
mses = np.zeros((len(lmbs), 1))
mses_OLS = np.zeros((len(lmbs), 1))
n_epochs = 10

I = np.zeros((3,3))
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1

#for j in range(len(n_epochs_list)):
#    n_epochs = n_epochs_list[j]
for j in range(len(lmbs)):
    lmb = lmbs[j]


    for epoch in range(n_epochs):
        for i in range(m):
            k = np.random.randint(m)
            X_k = X_train[k*M:(k+1)*M,:]
            y_k = y_train[k*M:(k+1)*M,:]
            #gradients = (2.0/M)* X_k.T @ ((X_k @ beta)-y_k)
            gradients = (2.0/M)* X_k.T @ ((X_k @ beta)-y_k) + lmb * I @ beta
            gamma = learning_schedule(epoch*m+i)
            beta_current = beta
            beta = beta - gamma*gradients




    beta_R= np.linalg.pinv(X_test.T @ X_test + lmb * I) @ X_test.T @ y_test


    y_R = X_test @ beta_R
    y_grad = X_test @ beta

    mse_grad = mean_squared_error(y_test, y_grad)
    mse_ols = mean_squared_error(y_test, y_R)

    mses[j]= mean_squared_error(y_test, y_grad)
    mses_OLS[j] = mean_squared_error(y_test, y_R)


plt.figure(1)
#plt.plot(n_epochs_list, mses, label = 'SGD MSEs')
#plt.plot(n_epochs_list, mses_OLS, label = 'Ridge MSE')
#plt.xlabel('Number of epochs $n_{epochs}$')
plt.plot(lmbs, mses, label = 'SGD MSEs')
plt.plot(lmbs, mses_OLS, label = 'Ridge MSE')
plt.xlabel('Penalty parameter $\lambda$')
plt.ylabel('MSE')
plt.legend()
plt.show()