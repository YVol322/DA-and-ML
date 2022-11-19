import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(1)

#2nd order polynomial we want to fit
n_inputs = 20
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

x = np.arange(0, 1, 1/n_inputs)
y = np.arange(0, 1, 1/n_inputs)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
y = z.reshape(-1,1)
x = x.reshape(-1,1)

n_inputs = n_inputs * n_inputs



#NN architecture
n_nodes = 15    #number of nodes in all hidden layers
n_layers = 3   #number of hidden layers
gamma = 0.1   #learning rate
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

#using sigmoid function as an activation function
def activation_function(x):
    return 1/(1+np.exp(-x))


#derrivative of sigmoid function
def sigmoid_prime(z):
        return np.multiply(activation_function(z), (1 - activation_function(z)))


#gradience of the cost function
def grad_C(a, y):
        return (a - y)


#Function that performs Feed Forwsard algo
def Feed_forward():
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
    




def Back_propagation():
    delta_L = grad_C(z_list[-1], y)
    delta_list.append(delta_L)

    for i in range(n_layers+1):
        delta_list.append(np.multiply(W_list[-1 - i].T @ delta_list[i], sigmoid_prime(z_list[-2 - i])))


    delta_list.reverse()

    for j in range(n_layers+2):
        W_list[-1-j] = W_list[-1-j] - gamma * delta_list[-1-j] @ a_list[-1-j].T
        b_list[-1-j] = b_list[-1-j] - gamma * delta_list[-1-j]
    


#rand_indexes = np.random.choice(n_inputs, int(n_inputs * 0.2))
#y_test = np.zeros((int(n_inputs * 0.2), 1))
#x_test = np.zeros((int(n_inputs * 0.2), 1))
#k=0
#
#W_1_test = np.random.randn(int(n_nodes/5), n_inputs)
#print(W_1.shape)
#for i in rand_indexes:
#    y_test[k] = y[i]
#    x_test[k] = x[i]
#
#    np.row
#    
#    
#    k+=1





j=0
while mean_squared_error(y, y_train_pred)>eps:
    Feed_forward()

    y_train_pred = z_list[-1]
    Back_propagation()
    print(mean_squared_error(y, y_train_pred))

    j+=1
    print(j)

plt.figure(1)
plt.title('$y = 2 + 3x + 4x^2 + N(0,0.3)$')
#plt.plot(x,y, 'ob', label = '$f(x) + noize$')
plt.plot(x,y_train_pred, 'k', label = 'grad fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()