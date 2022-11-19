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
n_nodes = 15    #number of nodes in all hidden layers
n_layers = 1   #number of hidden layers
gamma = 0.1   #learning rate
eps = 1e-8   #convergence criterior

#initialising the 1st hidden layer weights and biases

W_1 = np.random.randn(n_nodes, n_inputs)
b_1 = (np.zeros((n_nodes, 1)) + 0.01)


#initialising the 2nd, ..., L-1 hidden layer weights and biases

W_l = np.random.randn(n_layers-1, n_nodes, n_nodes)
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

for i in range(n_layers-1):
    W_list.append(W_l[i, :, :])
    b_list.append(b_l[i, :])

W_list.append(W_L)
b_list.append(b_L)

W_list_copy = []
b_list_copy = []
print(W_list[1].shape, b_list[1].shape)


#defining functions

#using sigmoid function as an activation function
def activation_function(x):
    return 1/(1+np.exp(-x))


#derrivative of sigmoid function
def sigmoid_prime(z):
        return np.multiply(activation_function(z), (1 - activation_function(z)))


#gradience of the cost function
def grad_C(a, y):
        return (a - y)


M = 100 #size of each mini bach
m = int(n_inputs/M) #number of minibatches
n_epochs = 50 #number of epochs

y_k = np.zeros((m,1))
y_pred_k = np.zeros((m,1))


#Function that performs Feed Forwsard algo
def Feed_forward(x_k):
    a_list.clear()
    z_list.clear()
    delta_list.clear()

    a_i = activation_function(x_k)

    a_list.append(a_i)
    z_1 = W_list_copy[0] @ a_i + b_list_copy[0]
    a_1 = activation_function(z_1)
    #print(W_list[1].shape, b_list[1].shape)
    z_list.append(z_1)
    a_list.append(a_1)


    for i in range(n_layers-1):
        z_list.append(W_list[i+1] @ a_list[i+1] + b_list[i+1].reshape(-1,1))
        a_list.append(activation_function(z_list[i+1]))

    z_L = W_list_copy[-1] @ a_list[-1] + b_list_copy[-1]
    z_list.append(z_L)
    




def Back_propagation(y_k):
    delta_L = grad_C(z_list[-1], y_k)
    delta_list.append(delta_L)

    for i in range(n_layers):
        delta_list.append(np.multiply(W_list[-1 - i].T @ delta_list[i], sigmoid_prime(z_list[-2 - i])))


    delta_list.reverse()

    for j in range(n_layers+1):
        W_list[-1-j] = W_list[-1-j] - gamma * delta_list[-1-j] @ a_list[-1-j].T
        b_list[-1-j] = b_list[-1-j] - gamma * delta_list[-1-j]
    
    




for epoch in range(n_epochs):
        for i in range(m):
            k = np.random.randint(m)
            x_k = x[k*M:(k+1)*M]
            y_k = y[k*M:(k+1)*M]
            W_list_copy = W_list
            b_list_copy = b_list
            Wi_k = W_list[0][:, k*M:(k+1)*M]
            WL_k = W_list[-1][k*M:(k+1)*M, :]
            bL_k = b_list[-1][k*M:(k+1)*M,:]
            W_list_copy[0] = Wi_k
            W_list_copy[-1] = WL_k
            b_list_copy[-1] = bL_k
            Feed_forward(x_k)
            y_pred_k = z_list[-1]
            Back_propagation(y_k)
            break
        break


plt.figure(1)
plt.title('$y = 2 + 3x + 4x^2 + N(0,0.3)$')
#plt.plot(x,y, 'ob', label = '$f(x) + noize$')
plt.plot(x,y_train_pred, 'k', label = 'grad fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()