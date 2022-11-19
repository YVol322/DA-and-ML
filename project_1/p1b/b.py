import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

x = np.arange(0, 1, 0.05)
y = x
X, Y = np.meshgrid(x, y)
z = FrankeFunction(x,y)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(X, Y, z, 'gray')

plt.show()