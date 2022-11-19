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


N = 20
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

maxdegree = 5

Beta = np.zeros((int((maxdegree+1)*(maxdegree+2)/2), maxdegree))

for i in range(1, maxdegree+1, 1):
    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)


    scaler = StandardScaler(with_std=False)
    X_scaled_train = scaler.fit_transform(X_train)
    z_scaled_train = scaler.fit_transform(z_train)

    beta_scaled = np.linalg.pinv(X_scaled_train.T @ (X_scaled_train)) @ (X_scaled_train.T) @ (z_scaled_train)


    for j in range(len(beta_scaled)):
        Beta[j, i-1] = beta_scaled[j]


    print('beta coeficients')
    print(Beta)



beta1 = Beta[0:3, 0]
beta2 = Beta[0:6, 1]
beta3 = Beta[0:10, 2]
beta4 = Beta[0:15, 3]
beta5 = Beta[:, 4]


degree1 = np.full(3, 1)
degree2 = np.full(6, 2)
degree3 = np.full(10, 3)
degree4 = np.full(15, 4)
degree5 = np.full(21, 5)


plt.figure(1)
plt.plot(degree1, beta1, 'ob', label = '1st degree,', )
plt.plot(degree2, beta2, 'ob', label = '2nd degree', color = 'r')
plt.plot(degree3, beta3, 'ob', label = '3rd degree', color = 'k')
plt.plot(degree4, beta4, 'ob', label = '4th degree', color = 'g')
plt.plot(degree5, beta5, 'ob', label = '5th degree', color = 'm')
plt.xlabel('Polynomial Fit Degree', fontsize = 15)
plt.ylabel('Fit coeficients', fontsize = 15)
plt.legend(fontsize = 10)
plt.show()