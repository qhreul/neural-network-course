import numpy as np
from neural_network_course.perceptron import Perceptron


class MultiLayerPerceptron:
    """
    A multilayer perceptron clkass that uses the Perceptron class
    Attributes:
        layers: List with the number of elements per layers
        bias: The bias term.
        eta: The learning rate
    """
    def __init__(self, layers: list, bias: float = 1.0, eta: float = 0.5):
        """
        :param layers: List with the number of elements per layers
        :param bias: The bias term. The default is 1.0.
        :param eta: the learning rate. The default is 0.5. 
        """
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        # The list of lists of neurons
        self.network = []
        # The list of lists of values
        self.values = []
        # The list of lists of error terms (lowercase deltas) 
        self.d = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    @property 
    def eta(self) -> float:
        """
        :return: The learning rate
        """
        return self._eta

    @eta.setter
    def eta(self, eta: float):
        """
        :param eta: The learning rate
        """
        self._eta = eta

    def set_weights(self, w_init):
        """
        Set the weights. 
        :pram w_init: A 3D list with the weights for all but the input layer.
        """
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].weights = w_init[i][j]

    def print_weights(self):
        """
        Print the weigths
        """
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)

    def run(self, input_values: list):
        """
        :param input_values
        """
        input_values = np.array(input_values, dtype=object)
        self.values[0] = input_values
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]

    def mean_squared_error(self, error) -> float:
        """
        :param error: 
        :return: The mean squard error
        """
        return sum(error ** 2) / self.layers[-1]

    def backprobagation(self, x, y):
        """
        :param x: Values
        :param y: Labels
        """
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        # step #1 feed a sample to the network
        outputs = self.run(x)

        # step #2 calculate the mean squared rate (MSE)
        error = y - outputs
        mse = self.mean_squared_error(error)

        # step #3 Calculate the output error terms
        self.d[-1] = outputs * (1 - outputs) * error

        # step #4 Calculate the error term for each unit on each hidden layer
        for i in reversed(range(1, len(self.network) - 1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i + 1][k].weights[h] * self.d[i+1][k]
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # step #5 and #6 Calculate the deltas and update the weights 
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]+1):
                    if k == self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        
        return mse

# test code
def training():
    mlp = MultiLayerPerceptron(layers=[2, 2, 1])  # mlp

    for i in range(3000):
        mse = 0.0
        mse += mlp.backprobagation([0, 0], [0])
        mse += mlp.backprobagation([0, 1], [1])
        mse += mlp.backprobagation([1, 0], [1])
        mse += mlp.backprobagation([1, 1], [0])
        mse = mse / 4
        if (i % 100 == 0):
            print(mse)

    mlp.print_weights()
    print("MLP:")
    print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
    print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
    print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
    print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))
