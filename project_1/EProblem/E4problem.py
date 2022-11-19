import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split

np.random.seed(2021)


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




def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n




def R2_Score_Function(y_predict, y_data):
    return (1 - np.sum((y_data - y_predict) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))





N = 20
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

degree = 5

nlambdas = 500
lambdas = np.logspace(-3, 7, nlambdas)

MSEs = np.zeros(nlambdas)

i = 0
for lmd in lambdas:

    X = create_X(x, y, degree)   


    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)  
    
    I = np.zeros((X_train.shape[1], X_train.shape[1]))
    np.fill_diagonal(I, lmd)


    beta = np.linalg.pinv(X_train.T @ (X_train) + I) @ (X_train.T) @ (z_train)  

    z_pred_test = X_test @ beta 

    MSEs[i] = MSE(z_pred_test, z_test)
    i += 1


plt.figure()

plt.plot(np.log10(lambdas), MSEs)

plt.xlabel('log10(lambda)', fontsize = 15)
plt.ylabel('Mean squeared error',fontsize = 15)

plt.legend(fontsize = 13)

plt.show()