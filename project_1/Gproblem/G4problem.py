import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from imageio import imread
from sklearn.utils import resample
from sklearn.model_selection import KFold



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

N = 200

terrain = terrain[:N,:N]

x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

x, y = np.meshgrid(x,y)

z = terrain
z = z.reshape(-1,1)

degree = 5

X = create_X(x, y, degree)

scaler = StandardScaler(with_std=False)
X_scaled = scaler.fit_transform(X)
z_scaled = scaler.fit_transform(z)


k = 10
kfold = KFold(n_splits = k)

scores_KFold_LR = np.zeros(k)
scores_KFold_R = np.zeros(k)
scores_KFold_L = np.zeros(k)

lmb = 0.1

i = 0
for train_inds, test_inds in kfold.split(x):
    X_train = X_scaled[train_inds]
    z_train = z_scaled[train_inds]

    X_test = X_scaled[test_inds]
    z_test = z_scaled[test_inds]

    I = np.zeros((X_train.shape[1], X_train.shape[1]))
    np.fill_diagonal(I, lmb)

    RegLasso = linear_model.Lasso(lmb)
    RegLasso.fit(X_train,z_train)
    
    beta_LR = np.linalg.pinv(X_train.T @ (X_train)) @ (X_train.T) @ (z_train)
    beta_R = np.linalg.pinv(X_train.T @ (X_train) + I) @ (X_train.T) @ (z_train)

    z_pred_LR = X_test @ beta_LR
    z_pred_R = X_test @ beta_R
    z_pred_L = RegLasso.predict(X_test)

    scores_KFold_LR[i] = np.mean((z_pred_LR - z_test)**2)
    scores_KFold_R[i] = np.mean((z_pred_R - z_test)**2)
    scores_KFold_L[i] = np.mean((z_pred_L - z_test)**2)

    i += 1

estimated_mse_KFold_LR = np.mean(scores_KFold_LR)
estimated_mse_KFold_R = np.mean(scores_KFold_R)
estimated_mse_KFold_L = np.mean(scores_KFold_L)

print(estimated_mse_KFold_LR)
print(estimated_mse_KFold_R)
print(estimated_mse_KFold_L)