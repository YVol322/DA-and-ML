from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import  StandardScaler
import numpy as np
from random import random, seed
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression

from Bproblem.Bproblem import MSE_test

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

#Fuction that calculates MSE
def Mean_Square_Error(y_predict, y_data):
    n = len(y_predict)
    return np.mean((y_data-y_predict)**2)

#Function that calculates R2
def R2_Score_Function(y_predict, y_data):
    return (1 - np.sum((y_data - y_predict) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))

# Make data.
N = 100
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

maxdegree = 5

MSE_test = np.zeros(maxdegree)
R2_test = np.zeros(maxdegree)

for i in range(1, maxdegree+1, 1):

	X = create_X(x, y, maxdegree)

	X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

	beta = np.linalg.inv(X_train.T @ (X_train)) @ (X_train.T) @ (z_train)

	z_pred_train = X_train @ beta
	z_pred_test = X_test @ beta

	MSE_test[i-1] = Mean_Square_Error(z_pred_test, z_test)
	R2_test[i-1] = R2_Score_Function(z_pred_test, z_test)

	print(MSE_test[i-1], R2_test[i-1])




#beta_scaled= np.linalg.pinv(X_train_scaled.T @ (X_train_scaled)) @ (X_train_scaled.T) @ (y_train)
#y_predict_train_scaled = X_train_scaled @ beta_scaled
#y_predict_test_scaled = X_test_scaled @ beta_scaled
#
#MSE_train_scaled = Mean_Square_Error(y_predict_test_scaled, y_test)
#R2_train_scaled = R2_Score_Function(y_predict_test_scaled, y_test)
#
#print(MSE_train_scaled, MSE_train)
#print(R2_train_scaled, R2_train)
#


# Plot the surface.

#z = z.reshape(100,100)
#
#
#print(x.shape, y.shape, z.shape)
#fig = plt.figure()
#ax = fig.gca(projection="3d")
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
## Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()