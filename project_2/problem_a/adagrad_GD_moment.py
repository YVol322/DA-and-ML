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
alpha = 0.5
Niterations = pow(10,6)
eps = pow(10,-8)

beta = np.random.randn(3,1)
betas = []
betas.append(beta)

G = np.zeros((3,3))


for i in range(Niterations):
    gradient = (2/n * X.T @ (X @ betas[i] - y))
    G = G + (gradient @ gradient.T)
    G_diag = G.diagonal()
    sqrt_G = np.sqrt(G_diag)
    gamma_k = gamma_0/(delta + sqrt_G).reshape(-1,1)
    if i == 0:
        betas.append(betas[i] - np.multiply(gamma_k, gradient))
    else:
        betas.append(betas[i] - gamma_k * gradient + alpha*(betas[i]-betas[i-1]))
    if abs(np.linalg.norm(betas[i+1]) - np.linalg.norm(betas[i])) < eps:
        print(i)
        break

beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta_ols)
print(betas[-1])

y_ols = X @ beta_ols
y_grad = X @ betas[-1]

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