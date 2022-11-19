import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(1)

n = 200
x = np.linspace(-1,1, n).reshape(-1,1)
y = 2 + 3 * x + 4 * np.multiply(x,x) + 0.3 * np.random.randn(n,1)

X = np.ones((n, 3))

for i in range(n):
    X[i, 1] = x[i]
    X[i, 2] = x[i] * x[i]

gamma_0 = 0.7
delta = pow(10,-7)
eps = pow(10,-8)

M = 5 #size of each mini bach
m = int(n/M) #number of minibatches
n_epochs = 50 #number of epochs

beta = np.random.randn(3,1)

G = np.zeros((3,3))


for epoch in range(n_epochs):
    for i in range(m):
        k = np.random.randint(m)
        X_k = X[k*M:(k+1)*M,:]
        y_k = y[k*M:(k+1)*M,:]
        gradients = (2.0/M)* X_k.T @ ((X_k @ beta)-y_k)
        G = G + (gradients @ gradients.T)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma_k = gamma_0/(delta + sqrt_G).reshape(-1,1)
        beta_current_iter = beta
        beta = beta - np.multiply(gamma_k, gradients)

beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y

print(beta_ols)
print(beta)

y_ols = X @ beta_ols
y_grad = X @ beta

mse_grad = mean_squared_error(y, y_grad)
mse_ols = mean_squared_error(y, y_ols)

print(mse_ols)
print(mse_grad)

plt.figure(1)
plt.title('$y = 2 + 3x + 4x^2 + N(0,0.3)$')
plt.plot(x,y, 'ob', label = '$f(x) + noize$')
plt.plot(x,y_grad, 'k', label = 'grad fit')
plt.plot(x,y_ols, 'r',linestyle ='--', label = 'MSE fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()