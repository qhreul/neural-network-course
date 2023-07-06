import numpy as np

class Perceptron:
    """
    A single neuron with the sigmoid activation function
    Attributes:
        inputs: The number of inputs in the perceptron, not counting the bias
        bias: The bias term. By default, it is 1.0
    """
    def __init__(self, inputs: int, bias: float = 1.0):
        """
        :oaram inputs: The number of inputs in the perceptron
        :param bias: the bias term
        :return: a new Perceptron object with the specified number of inputs
        """
        self.weights = (np.random.rand(inputs+1) * 2) - 1
        self.bias = bias

    @property
    def bias(self) -> float:
        """
        :return: the bias term as a float
        """
        return self._bias
    
    @bias.setter
    def bias(self, bias: float):
        """
        :param bias: The bias term as a float
        """
        self._bias = bias
    
    @property
    def weights(self) -> list:
        """
        :return: a list of floats as weigths
        """
        return self._weights

    @weights.setter
    def weights(self, weights: list[int]):
        """
        :param weights: is a list of floats
        """
        self._weights = np.array(weights)

    def sigmoid(self, x):
        """
        :param x: 
        :return: The output from the sigmoid function applied to x 
        """
        return 1 / ( 1 + np.exp(-x) )

    def run(self, input_values: list):
        """
        Run the perceptron
        :param input_values: A list with input values
        :return: A signmoid function
        """
        x_sum = np.dot(np.append(input_values, self.bias), self.weights)
        return self.sigmoid(x_sum)
