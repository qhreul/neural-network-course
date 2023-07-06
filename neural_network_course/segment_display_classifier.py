import numpy as np
from neural_network_course.multilayer_perceptron import MultiLayerPerceptron

def training_sdc_7by7(epochs: int):
    mlp = MultiLayerPerceptron(layers=[7, 7, 7])

    for i in range(epochs):
        mse = 0.0
        mse += mlp.backprobagation([1,1,1,1,1,1,0],[1,1,1,1,1,1,0])    #0 pattern
        mse += mlp.backprobagation([0,1,1,0,0,0,0],[0,1,1,0,0,0,0])    #1 pattern
        mse += mlp.backprobagation([1,1,0,1,1,0,1],[1,1,0,1,1,0,1])    #2 pattern
        mse += mlp.backprobagation([1,1,1,1,0,0,1],[1,1,1,1,0,0,1])    #3 pattern
        mse += mlp.backprobagation([0,1,1,0,0,1,1],[0,1,1,0,0,1,1])    #4 pattern
        mse += mlp.backprobagation([1,0,1,1,0,1,1],[1,0,1,1,0,1,1])    #5 pattern
        mse += mlp.backprobagation([1,0,1,1,1,1,1],[1,0,1,1,1,1,1])    #6 pattern
        mse += mlp.backprobagation([1,1,1,0,0,0,0],[1,1,1,0,0,0,0])    #7 pattern
        mse += mlp.backprobagation([1,1,1,1,1,1,1],[1,1,1,1,1,1,1])    #8 pattern
        mse += mlp.backprobagation([1,1,1,1,0,1,1],[1,1,1,1,0,1,1])    #9 pattern
        mse = mse/10.0

        if (i % 500 == 0):
            print(mse)

    return mlp


def training_sdc_7by10(epochs: int):
    mlp = MultiLayerPerceptron(layers=[7, 7, 10])

    for i in range(epochs):
        mse = 0.0
        mse += mlp.backprobagation([1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0,0,0])    #0 pattern
        mse += mlp.backprobagation([0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0])    #1 pattern
        mse += mlp.backprobagation([1,1,0,1,1,0,1],[0,0,1,0,0,0,0,0,0,0])    #2 pattern
        mse += mlp.backprobagation([1,1,1,1,0,0,1],[0,0,0,1,0,0,0,0,0,0])    #3 pattern
        mse += mlp.backprobagation([0,1,1,0,0,1,1],[0,0,0,0,1,0,0,0,0,0])    #4 pattern
        mse += mlp.backprobagation([1,0,1,1,0,1,1],[0,0,0,0,0,1,0,0,0,0])    #5 pattern
        mse += mlp.backprobagation([1,0,1,1,1,1,1],[0,0,0,0,0,0,1,0,0,0])    #6 pattern
        mse += mlp.backprobagation([1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0])    #7 pattern
        mse += mlp.backprobagation([1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,0])    #8 pattern
        mse += mlp.backprobagation([1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0,0,1])    #9 pattern
        mse = mse/10.0

        if (i % 500 == 0):
            print(mse)

    return mlp
