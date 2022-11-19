import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from imageio import imread
from sklearn.utils import resample



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

N = 250

maxdegree = 5

terrain = terrain[:N,:N]

x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

x, y = np.meshgrid(x,y)

z = terrain
z = z.reshape(-1,1)

MSE_LR = np.zeros(maxdegree)
BIAS_LR = np.zeros(maxdegree)
VAR_LR = np.zeros(maxdegree)

MSE_R = np.zeros(maxdegree)
BIAS_R = np.zeros(maxdegree)
VAR_R = np.zeros(maxdegree)

MSE_L = np.zeros(maxdegree)
BIAS_L = np.zeros(maxdegree)
VAR_L = np.zeros(maxdegree)

polydegree = np.zeros(maxdegree)

lmb = 0.1

n_boostraps = 20

for i in range(1, maxdegree+1, 1):

    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    scaler = StandardScaler(with_std=False)
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.fit_transform(X_test)
    z_scaled_train = scaler.fit_transform(z_train)
    z_scaled_test = scaler.fit_transform(z_test)

    z_pred_LR = np.empty((z_test.shape[0], n_boostraps))
    z_pred_R = np.empty((z_test.shape[0], n_boostraps))
    z_pred_L = np.empty((z_test.shape[0], n_boostraps))


    for j in range(n_boostraps):
        X_, z_ = resample(X_scaled_train, z_scaled_train)

        I = np.zeros((X_train.shape[1], X_train.shape[1]))
        np.fill_diagonal(I, lmb)

        RegLasso = linear_model.Lasso(lmb)
        RegLasso.fit(X_scaled_train,z_scaled_train)

        beta_LR = np.linalg.pinv(X_.T @ (X_)) @ (X_.T) @ (z_)
        beta_R = np.linalg.pinv(X_.T @ (X_) + I) @ (X_.T) @ (z_)

        z_pred_LR[:, j] = (X_scaled_test @ beta_LR).ravel()
        z_pred_R[:, j] = (X_scaled_test @ beta_R).ravel()
        z_pred_L[:, j] = RegLasso.predict(X_scaled_test).ravel()


    MSE_LR[i-1] = np.mean( np.mean((z_test - z_pred_LR)**2, axis=1, keepdims=True) )
    BIAS_LR[i-1] = np.mean( (z_test - np.mean(z_pred_LR, axis=1, keepdims=True))**2 )
    VAR_LR[i-1] = np.mean( np.var(z_pred_LR, axis=1, keepdims=True) )

    MSE_R[i-1] = np.mean( np.mean((z_test - z_pred_R)**2, axis=1, keepdims=True) )
    BIAS_R[i-1] = np.mean( (z_test - np.mean(z_pred_R, axis=1, keepdims=True))**2 )
    VAR_R[i-1] = np.mean( np.var(z_pred_R, axis=1, keepdims=True) )

    MSE_L[i-1] = np.mean( np.mean((z_test - z_pred_L)**2, axis=1, keepdims=True) )
    BIAS_L[i-1] = np.mean( (z_test - np.mean(z_pred_L, axis=1, keepdims=True))**2 )
    VAR_L[i-1] = np.mean( np.var(z_pred_L, axis=1, keepdims=True) )

    polydegree[i-1] = i

    print('LR Error:', MSE_LR[i-1])
    print('LR Bias^2:', BIAS_LR[i-1])
    print('LR Var:', VAR_LR[i-1])
    print('{} >= {} + {} = {}'.format(MSE_LR[i-1], BIAS_LR[i-1], VAR_LR[i-1], VAR_LR[i-1]+BIAS_LR[i-1]))

    print('R Error:', MSE_R[i-1])
    print('R Bias^2:', BIAS_R[i-1])
    print('R Var:', VAR_R[i-1])
    print('{} >= {} + {} = {}'.format(MSE_R[i-1], BIAS_R[i-1], VAR_R[i-1], VAR_R[i-1]+BIAS_R[i-1]))

    print('L Error:', MSE_R[i-1])
    print('L Bias^2:', BIAS_R[i-1])
    print('L Var:', VAR_R[i-1])
    print('{} >= {} + {} = {}'.format(MSE_L[i-1], BIAS_L[i-1], VAR_L[i-1], VAR_L[i-1]+BIAS_L[i-1]))

plt.figure(1)
plt.plot(polydegree, MSE_LR, label = 'Linear regression MSE')
plt.plot(polydegree, BIAS_LR, label = 'Linear regression BIAS^2')
plt.plot(polydegree, VAR_LR, label = 'Linear regression Varianse')
plt.xlabel("Polynomial fit degree", fontsize = 15)
plt.ylabel("Error value", fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(2)
plt.plot(polydegree, MSE_R, label = 'Ridge regression MSE')
plt.plot(polydegree, BIAS_R, label = 'Ridge regression BIAS^2')
plt.plot(polydegree, VAR_R, label = 'Ridge regression Varianse')
plt.xlabel("Polynomial fit degree", fontsize = 15)
plt.ylabel("Error value", fontsize = 15)
plt.legend(fontsize = 13)

plt.figure(3)
plt.plot(polydegree, MSE_L, label = 'Lasso regression MSE')
plt.plot(polydegree, BIAS_L, label = 'Lasso regression BIAS^2')
plt.plot(polydegree, VAR_L, label = 'Lasso regression Varianse')
plt.xlabel("Polynomial fit degree", fontsize = 15)
plt.ylabel("Error value", fontsize = 15)
plt.legend(fontsize = 13)
plt.show()