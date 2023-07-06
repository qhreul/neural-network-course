import numpy as np
import pytest
from neural_network_course import segment_display_classifier

@pytest.mark.parametrize('epochs, pattern, output', [
    (1000, [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0])])
def test_training_sdc_7by7(epochs: int, pattern: list, output: int):
    mlp = segment_display_classifier.training_sdc_7by7(epochs)
    prediction = mlp.run(pattern)

    assert [int(x) for x in (prediction + 0.5)] == output
    
    
@pytest.mark.parametrize('epochs, pattern, output', [
    (1000, [1, 1, 1, 1, 1, 1, 0], 0),
    (2000, [0.1, 0.8, 0.8, 0.005, 0, 0, 0.2], 1)])
def test_training_sdc_7by10(epochs: int, pattern: list, output: int):
    mlp = segment_display_classifier.training_sdc_7by10(epochs)
    prediction = mlp.run(pattern)

    assert np.argmax(prediction) == output
