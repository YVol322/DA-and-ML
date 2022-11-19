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
delta = pow(10,-5)
Niterations = pow(10,6)
eps = pow(10,-8)

beta = np.random.randn(3,1)

first_moment = 0.0
second_moment = 0.0

beta1 = 0.99
beta2 = 0.999

for i in range(Niterations):
    gradient = (2/n * X.T @ (X @ beta - y))
    first_moment = beta1*first_moment + (1-beta1)*gradient
    second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
    first_term = first_moment/(1.0-beta1**i + delta)
    second_term = second_moment/(1.0-beta2**i + delta)
    beta_current_iter = beta
    beta = beta - gamma_0*first_term/(np.sqrt(second_term)+delta)
    if abs(np.linalg.norm(beta) - np.linalg.norm(beta_current_iter)) < eps:
        print(i)
        break

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
plt.show()