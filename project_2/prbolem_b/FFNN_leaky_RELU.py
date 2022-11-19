import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(1)

#2nd order polynomial we want to fit
n_inputs = 10000
x = np.linspace(-1,1, n_inputs).reshape(-1,1)
y = 2 + 3 * x + 4 * np.multiply(x,x) #+ 0.3 * np.random.randn(n_inputs,1)



#NN architecture
n_nodes = 15   #number of nodes in all hidden layers
n_layers = 5   #number of hidden layers
gamma = 0.9   #learning rate
eps = 1e-8   #convergence criterior

#initialising the 1st hidden layer weights and biases

W_1 = np.random.randn(n_nodes, n_inputs)
b_1 = (np.zeros((n_nodes, 1)) + 0.01)


#initialising the 2nd, ..., L-1 hidden layer weights and biases

W_l = np.random.randn(n_layers, n_nodes, n_nodes)
b_l = (np.zeros((n_layers ,n_nodes, 1)) + 0.01)

#initialising the output layer weights and biases

W_L = np.random.randn(n_inputs, n_nodes)
b_L = (np.zeros((n_inputs, 1)) + 0.01)



#initilising some data

y_train_pred = np.zeros((y.shape))   #initilising the prediction of NN

z_list = []   #list of input to the layes
a_list = []   #list of output of the layers

W_list = []   #list of weights of 2, ... , L layers
b_list = []   #list of biases of 2, ... , L layers

delta_list = []   ##list of deltas

W_list.append(W_1)
b_list.append(b_1)

for i in range(n_layers):
    W_list.append(W_l[i, :, :])
    b_list.append(b_l[i, :])

W_list.append(W_L)
b_list.append(b_L)





#defining functions

#using RELU function as an activation function
def activation_function(x):
    R = np.zeros((x.shape))
    for i in range(x.shape[0]):
        if x[i] > 0.1 * x[i]:
            R[i] = x[i]
        else:
            R[i] = 0.1 * x[i]
        return R


#derrivative of RELU function
def RELU_prime(z):
        R = np.zeros((z.shape))
        for i in range(z.shape[0]):
            if z[i] > 0.1 * z[i]:
                R[i] = 1
            else:
                R[i] = 0.1
            return R
                


#gradience of the cost function
def grad_C(a, y):
        return 2/n_inputs * (a - y)


#Function that performs Feed Forwsard algo
def Feed_forward(x):
    a_list.clear()
    z_list.clear()
    delta_list.clear()

    a_i = activation_function(x)

    a_list.append(a_i)


    z_1 = W_list[0] @ a_i + b_list[0]
    a_1 = activation_function(z_1)
    


    z_list.append(z_1)
    a_list.append(a_1)


    for i in range(n_layers):
        z_list.append(W_list[i+1] @ a_list[i+1] + b_list[i+1].reshape(-1,1))
        a_list.append(activation_function(z_list[i+1]))

    z_L = W_list[-1] @ a_list[-1] + b_list[-1]
    z_list.append(z_L)
    




def Back_propagation(y):
    delta_L = grad_C(z_list[-1], y)
    delta_list.append(delta_L)

    for i in range(n_layers+1):
        delta_list.append(np.multiply(W_list[-1 - i].T @ delta_list[i], RELU_prime(z_list[-2 - i])))


    delta_list.reverse()

    for j in range(n_layers+2):
        W_list[-1-j] = W_list[-1-j] - gamma * 2/n_inputs * delta_list[-1-j] @ a_list[-1-j].T #  - lmd *
        b_list[-1-j] = b_list[-1-j] - gamma *2/n_inputs* delta_list[-1-j]



j=0
while mean_squared_error(y, y_train_pred)>eps:
    Feed_forward(x)

    y_train_pred = z_list[-1]
    Back_propagation(y)
    print(mean_squared_error(y, y_train_pred))

    j+=1
    print(j)


noise = 0.3 * np.random.randn(n_inputs, 1)
y_noise = y + noise

Feed_forward(x)
y_train_pred = z_list[-1]
print(mean_squared_error(y_noise, y_train_pred))
print(mean_squared_error(y, y_train_pred))

plt.figure(1)
plt.plot(x,y_noise, 'ob', label = '$f(x) + noize$')
plt.plot(x,y_train_pred, 'k', label = 'FFNN fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
