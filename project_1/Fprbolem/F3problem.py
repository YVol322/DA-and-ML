import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model

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
	l = int((n+1)*(n+2)/2)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X




N = 400
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

degree = 5

X = create_X(x, y, degree)

k = 5
kfold = KFold(n_splits = k)

scores_KFold = np.zeros(k)

lmb = 1

i = 0
for train_inds, test_inds in kfold.split(x):
    X_train = X[train_inds]
    z_train = z[train_inds]

    X_test = X[test_inds]
    z_test = z[test_inds]

    RegLasso = linear_model.Lasso(lmb)
    RegLasso.fit(X_train,z_train)
    z_pred = RegLasso.predict(X_test)

    scores_KFold[i] = np.mean((z_pred - z_test)**2)

    i += 1

estimated_mse_KFold = np.mean(scores_KFold)

print(estimated_mse_KFold)