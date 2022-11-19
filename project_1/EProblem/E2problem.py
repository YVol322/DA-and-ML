import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.utils import resample

np.random.seed(2022)


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




# Make data.
N = 10
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

maxdegree = 5
n_boostraps = 100


polydegree = np.zeros(maxdegree)

MSE = np.zeros(maxdegree)
BIAS = np.zeros(maxdegree)
VAR = np.zeros(maxdegree)

for i in range(1, maxdegree+1, 1):

    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    z_pred = np.empty((z_test.shape[0], n_boostraps))

    lmbda = 1
    I = np.zeros((X_train.shape[1], X_train.shape[1]))
    np.fill_diagonal(I, lmbda)


    for j in range(n_boostraps):
        X_, z_ = resample(X_train, z_train)

        beta = np.linalg.pinv(X_.T @ (X_) + I) @ (X_.T) @ (z_)

        z_pred[:, j] = (X_test @ beta).ravel()

    MSE[i-1] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    BIAS[i-1] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    VAR[i-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    
    polydegree[i-1] = i

    print('Error:', MSE[i-1])
    print('Bias^2:', BIAS[i-1])
    print('Var:', VAR[i-1])
    print('{} >= {} + {} = {}'.format(MSE[i-1], BIAS[i-1], VAR[i-1], VAR[i-1]+BIAS[i-1]))

plt.figure()
plt.plot(polydegree, MSE, label = 'MSE')
plt.plot(polydegree, BIAS, label = 'BIAS^2')
plt.plot(polydegree, VAR, label = 'Varianse')
plt.xlabel("Polynomial fit degree", fontsize = 15)
plt.ylabel("Error value", fontsize = 15)
plt.legend(fontsize = 13)
plt.show()