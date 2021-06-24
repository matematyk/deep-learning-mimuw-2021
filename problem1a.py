import random
import numpy as np
from torchvision import datasets, transforms

# Let's read the mnist dataset

def load_mnist(path='.'):
    from torchvision.datasets import MNIST

    train_set = MNIST(root = './', train=True, download=True)
    test_set = MNIST(root = './', train=False, download=True)

    x_train = train_set.data.numpy()
    _y_train = train_set.targets.numpy()
    
    x_test = test_set.data.numpy()
    _y_test = test_set.targets.numpy()
    
    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = np.zeros((_y_train.shape[0], 10))
    y_train[np.arange(_y_train.shape[0]), _y_train] = 1
    
    y_test = np.zeros((_y_test.shape[0], 10))
    y_test[np.arange(_y_test.shape[0]), _y_test] = 1

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist()


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        # initialize biases and weights with random normal distr.
        # weights are indexed by target node first
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, a):
        # Run the network on a single case
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def update_mini_batch(self, x_mini_batch, y_mini_batch, eta):
        # Update networks weights and biases by applying a single step
        # of gradient descent using backpropagation to compute the gradient.
        # The gradient is computed for a mini_batch.
        # eta is the learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(x_mini_batch, y_mini_batch):
            delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(784,1), y.reshape(10,1)) #wywołanie 
            #dL/dw
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(x_mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(x_mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
        # print(self.weights)
        
    def backprop(self, x, y):
        # For a single input (x,y) return a tuple of lists.
        # First contains gradients over biases, second over weights.
        
        # First initialize the list of gradient arrays
        delta_nabla_b = []
        delta_nabla_w = []
        
        # Then go forward remembering all values before and after activations
        # in two other array lists

        gs = [x]
        fs = []
        for b, w in zip(self.biases, self.weights):
            fs.append(np.dot(w, gs[-1]) + b) #poprzednia wartosc + biase
            gs.append(sigmoid(fs[-1]))

        # Now go backward from the final cost applying backpropagation

        dg = gs[-1] - y #pochodna MSE 2krotonosci Dl/dg
        dfs = []
        #idac od tylu
        for w, g, f in reversed(list(zip(self.weights, gs[1:], fs))):
            dfs.append(np.multiply(dg, sigmoid_prime(f))) #DL/df
            dg = np.matmul(w.T, dfs[-1]) #update Dl/dg^{k-1}= W^T Dl/df^{k}
        #wszystkie pochodne F
        for df, g, w in zip(reversed(dfs), gs[:-1], self.weights):
        #dR/dw_i = e^w_i / (e^w1 + … + e^w_n) @TODO
          licznik = np.exp(w)
          mianownik = np.sum(licznik)
          delta_nabla_w.append(np.matmul(df, g.T)+licznik/mianownik) #dL/dW^k = Dl/df^k(g^{k-1})^T
          delta_nabla_b.append(np.sum(df, axis=1)[:, np.newaxis]) 
        #Dl/Dw =delta_nabla_w
        return delta_nabla_b, delta_nabla_w

    def evaluate(self, x_test_data, y_test_data):
        # Count the number of correct answers for test_data
        test_results = [(np.argmax(self.feedforward(x_test_data[i].reshape(784,1))), np.argmax(y_test_data[i]))
                        for i in range(len(x_test_data))]
        # return accuracy
        return np.mean([int(x == y) for (x, y) in test_results])
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y) 
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
            # for i in range(1):
                x_mini_batch = x_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)] 
                y_mini_batch = y_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)] 
                self.update_mini_batch(x_mini_batch, y_mini_batch, eta)
            if test_data:
                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate(x_test, y_test)))
            else:
                print("Epoch: {0}".format(j))


network = Network([784,30,10])
network.SGD((x_train, y_train), epochs=50, mini_batch_size=100, eta=3., test_data=(x_test, y_test))

