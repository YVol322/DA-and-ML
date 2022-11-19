import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn.model_selection import  train_test_split

np.random.seed(3)



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X



def Mean_Square_Error(y_predict, y_data):
    return np.mean((y_data-y_predict)**2)



N = 20
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

maxdegree = 11

MSE_scaled_test = np.zeros(maxdegree)
MSE_scaled_train = np.zeros(maxdegree)

polydegree = np.zeros(maxdegree)


for i in range(1, maxdegree+1, 1):
    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)


    scaler = StandardScaler(with_std=False)
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.fit_transform(X_test)
    z_scaled_train = scaler.fit_transform(z_train)
    z_scaled_test = scaler.fit_transform(z_test)

    beta_scaled = np.linalg.pinv(X_scaled_train.T @ (X_scaled_train)) @ (X_scaled_train.T) @ (z_scaled_train)

    z_scaled_pred_train = X_scaled_train @ beta_scaled
    z_scaled_pred_test = X_scaled_test @ beta_scaled


    MSE_scaled_test[i-1] = Mean_Square_Error(z_scaled_pred_test, z_scaled_test)
    MSE_scaled_train[i-1] = Mean_Square_Error(z_scaled_pred_train, z_scaled_train)


    polydegree[i-1] = i


    print('polynomial fit degree', '\n', i)
    print('Train, Test MSE')
    print(MSE_scaled_train[i-1], MSE_scaled_test[i-1])



plt.figure(1)
plt.plot(polydegree, MSE_scaled_test, label='Test MSE for scaled data')
plt.plot(polydegree, MSE_scaled_train, label='Train MSE for scaled data')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)
plt.show()