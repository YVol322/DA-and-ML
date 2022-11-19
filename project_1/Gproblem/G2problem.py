import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from imageio import imread
from sklearn.metrics import mean_squared_error, r2_score



np.random.seed(3)



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



terrain = imread('SRTM_data_Norway_1.tif')

N = 1000

maxdegree = 5

terrain = terrain[:N,:N]

x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

x, y = np.meshgrid(x,y)

z = terrain
z = z.reshape(-1,1)

MSE_test_LR = np.zeros(maxdegree)
MSE_train_LR = np.zeros(maxdegree)

MSE_test_R = np.zeros(maxdegree)
MSE_train_R = np.zeros(maxdegree)

MSE_test_L = np.zeros(maxdegree)
MSE_train_L = np.zeros(maxdegree)

R2_test_LR = np.zeros(maxdegree)
R2_train_LR = np.zeros(maxdegree)

R2_test_R = np.zeros(maxdegree)
R2_train_R = np.zeros(maxdegree)

R2_test_L = np.zeros(maxdegree)
R2_train_L = np.zeros(maxdegree)


polydegree = np.zeros(maxdegree)

lmb = 0.1


for i in range(1, maxdegree+1, 1):
    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)


    scaler = StandardScaler(with_std=False)
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.fit_transform(X_test)
    z_scaled_train = scaler.fit_transform(z_train)
    z_scaled_test = scaler.fit_transform(z_test)

    I = np.zeros((X_scaled_train.shape[1], X_scaled_train.shape[1]))
    np.fill_diagonal(I, lmb)

    RegLasso = linear_model.Lasso(lmb)
    RegLasso.fit(X_scaled_train,z_scaled_train)

    beta_LR = np.linalg.pinv(X_scaled_train.T @ (X_scaled_train)) @ (X_scaled_train.T) @ (z_scaled_train)
    beta_R = np.linalg.pinv(X_scaled_train.T @ (X_scaled_train) + I) @ (X_scaled_train.T) @ (z_scaled_train)

    z_pred_train_LR = X_scaled_train @ beta_LR
    z_pred_test_LR = X_scaled_test @ beta_LR

    z_pred_train_R = X_scaled_train @ beta_R
    z_pred_test_R = X_scaled_test @ beta_R

    z_pred_train_L = RegLasso.predict(X_scaled_train)
    z_pred_test_L = RegLasso.predict(X_scaled_test)


    MSE_test_LR[i-1] = mean_squared_error(z_scaled_test, z_pred_test_LR)
    MSE_train_LR[i-1] = mean_squared_error(z_scaled_train, z_pred_train_LR)

    MSE_test_R[i-1] = mean_squared_error(z_scaled_test, z_pred_test_R)
    MSE_train_R[i-1] = mean_squared_error(z_scaled_train, z_pred_train_R)

    MSE_test_L[i-1] = mean_squared_error(z_scaled_test, z_pred_test_L)
    MSE_train_L[i-1] = mean_squared_error(z_scaled_train, z_pred_train_L)

    R2_test_LR[i-1] = r2_score(z_scaled_test, z_pred_test_LR)
    R2_train_LR[i-1] = r2_score(z_scaled_train, z_pred_train_LR)

    R2_test_R[i-1] = r2_score(z_scaled_test, z_pred_test_R)
    R2_train_R[i-1] = r2_score(z_scaled_train, z_pred_train_R)

    R2_test_L[i-1] = r2_score(z_scaled_test, z_pred_test_L)
    R2_train_L[i-1] = r2_score(z_scaled_train, z_pred_train_L)
    


    polydegree[i-1] = i


    print('polynomial fit degree', '\n', i)

    print('Linear, Ridge, Lasso regression train MSE')
    print(MSE_train_LR[i-1], MSE_train_R[i-1], MSE_train_L[i-1])

    print('Linear, Ridge, Lasso regression test MSE')
    print(MSE_test_LR[i-1], MSE_test_R[i-1], MSE_test_L[i-1])

    print('Linear, Ridge, Lasso regression train R2 score')
    print(R2_train_LR[i-1], R2_train_R[i-1], R2_train_L[i-1])

    print('Linear, Ridge, Lasso regression test R2 score')
    print(R2_test_LR[i-1], R2_test_R[i-1], R2_test_L[i-1])
    


plt.figure(1)
plt.plot(polydegree, MSE_test_LR, label='Linear regression test MSE')
plt.plot(polydegree, MSE_test_R, label='Ridge regression test MSE')
plt.plot(polydegree, MSE_test_L, label='Lasso regression test MSE')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)


plt.figure(2)
plt.plot(polydegree, R2_test_LR, label='Linear regression test R2')
plt.plot(polydegree, R2_test_R, label='Ridge regression test R2')
plt.plot(polydegree, R2_test_L, label='Lasso regression test R2')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('R2 Score Function', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(3)
plt.plot(polydegree, MSE_train_LR, label='Linear regression train MSE')
plt.plot(polydegree, MSE_test_LR, label='Linear regression test MSE')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(4)
plt.plot(polydegree, MSE_train_R, label='Ridge regression train MSE')
plt.plot(polydegree, MSE_test_R, label='Ridge regression test MSE')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(5)
plt.plot(polydegree, MSE_train_L, label='Lasso regression train MSE')
plt.plot(polydegree, MSE_test_L, label='Lasso regression test MSE')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(6)
plt.plot(polydegree, R2_train_LR, label='Linear regression train R2')
plt.plot(polydegree, R2_test_LR, label='Linear regression test R2')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(7)
plt.plot(polydegree, R2_train_R, label='Ridge regression train R2')
plt.plot(polydegree, R2_test_R, label='Ridge regression test R2')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(8)
plt.plot(polydegree, R2_train_L, label='Lasso regression train R2')
plt.plot(polydegree, R2_test_L, label='Lasso regression test R2')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Mean Squared Error', fontsize = 15)
plt.legend(fontsize = 13)
plt.show()