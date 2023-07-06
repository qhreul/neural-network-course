"""
Module to test the Perceptron class
"""
import pytest
from neural_network_course.perceptron import Perceptron

@pytest.mark.parametrize('bias', [0.2, 0.3, 0.5, 0.6])
def test_init(bias: float):
    """
    Test the __init__ method for the Perceptron object
    :param bias: The bias term as a float
    """
    neuron = Perceptron(inputs=2, bias=bias)

    assert neuron.bias == bias


@pytest.mark.parametrize('combination, score', [([0, 0], 3.059022269256247e-07),
                                                ([0, 1], 0.0066928509242848554),
                                                ([1, 0], 0.0066928509242848554),
                                                ([1, 1], 0.9933071490757153)])
def test_run_and_gate(combination: list, score: float):
    """
    :param combination: combination of values for the Perceptron object
    :param score: score of run() function for the combination
    """          
    neuron = Perceptron(inputs=2)
    neuron.weights = [10, 10, -15]

    assert neuron.run(combination) == score


@pytest.mark.parametrize('combination, score', [([0, 0], 4.5397868702434395e-05),
                                                ([0, 1], 0.9933071490757153),
                                                ([1, 0], 0.9933071490757153),
                                                ([1, 1], 0.9999999979388463)])
def test_run_or_gate(combination: list, score: float):
    neuron = Perceptron(inputs=2)
    neuron.weights = [15, 15, -10]

    assert neuron.run(combination) == score
