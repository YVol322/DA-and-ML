from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

zNoise = z + np.random.normal(0,0.1, size=z.shape)
## Plot the surface.
#surf = ax.plot_surface(x, y, zNoise, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
## Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
#
#1st degree
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X = np.zeros((len(x), 3))
X[:, 0] = 1.0
X[:, 1] = x
X[:, 2] = y

X_train, X_test, z_train, z_test, x_train, x_test, y_train, y_test= train_test_split(X, zNoise, x, y, test_size=0.2)

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

ztilde = X_train @ beta

x_train, y_train = np.meshgrid(x_train,y_train)

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(x_train, y_train, ztilde, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#def Plot_Franke_Function(x, y, z):
#    x, y = np.meshgrid(x, y)
#    z = np.meshgrid(ztilde)
#    fig = plt.figure()
#    ax = fig.gca(projection="3d")
#    # Plot the surface.
#    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    # Customize the z axis.
#    ax.set_zlim(-0.10, 1.40)
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.show()

#x, y = np.meshgrid(x, y)
#x_Noize = np.random.normal(0, 0.05, size=z.shape)
#y_Noize = np.random.normal(0, 0.05, size=z.shape)
#x_Noize, y_Noize = np.meshgrid(x_Noize, y_Noize)
#z_Noize = FrankeFunction(x, y) + x_Noize + y_Noize

#plt.figure(1)
#plt.plot(degree, MSE_train, "ob", color = "r", label = "Train")
#plt.plot(degree, MSE_test, "ob", color = "k", label = "Test")
#plt.xlabel('Polynomial fit degree')
#plt.ylabel('Mean square error')
#plt.legend()
#plt.show()

    X_train, X_test, z_train, z_test = train_test_split(X,z_noize,test_size=0.2)
    
    
    beta = np.linalg.inv(X_train.T @ (X_train)) @ (X_train.T) @ (z_train)
    zpredict_train = X_train @ beta
    zpredict_test = X_test @ beta

    MSE_train, MSE_test, R_2_train, R_2_test = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    
    MSE_train[i-1] = Mean_Square_Error(zpredict_train, z_train)
    MSE_test[i-1] = Mean_Square_Error(zpredict_test, z_test)
    R_2_train[i-1] = R_2_Score_Function(zpredict_train, z_train)
    R_2_test[i-1] = R_2_Score_Function(zpredict_test, z_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    beta_scaled = np.linalg.pinv(X_train_scaled.T @ (X_train_scaled)) @ (X_train_scaled.T) @ (z_train)
    zpredict_train_scaled = X_train_scaled @ beta_scaled
    zpredict_test_scaled = X_test_scaled @ beta_scaled

    MSE_train_scaled, MSE_test_scaled, R_2_train_scaled, R_2_test_scaled = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    
    MSE_train_scaled[i-1] = Mean_Square_Error(zpredict_train_scaled, z_train)
    MSE_test_scaled[i-1] = Mean_Square_Error(zpredict_test_scaled, z_test)
    R_2_train_scaled[i-1] = R_2_Score_Function(zpredict_train_scaled, z_train)
    R_2_test_scaled[i-1] = R_2_Score_Function(zpredict_test_scaled, z_test)

    print(i)
    print("Train MSE", '\n', MSE_train[i-1], '\n', "Scaled Train MSE", '\n', MSE_train_scaled[i-1])
    print("Test MSE", '\n', MSE_test[i-1], '\n', "Scaled Test MSE", '\n', MSE_test_scaled[i-1])
    print("Train R_2", '\n', R_2_train[i-1], '\n', "Scaled Train R_2", '\n', R_2_train_scaled[i-1])
    print("Test R_2", '\n', R_2_test[i-1], '\n', "Scaled Test R_2", '\n', R_2_test_scaled[i-1])


degree = np.arange(1,6, 1)
