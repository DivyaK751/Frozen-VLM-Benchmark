import torch
import torch.nn as nn

from small_vlm.connector import pool_visual_tokens, VisualProjector

def test_pooling_shape():
    """ verify pooling operation returns the correct shape"""

    tokens = torch.randn(2, 729, 1152)
    pooling_output = pool_visual_tokens(tokens)
    assert pooling_output.shape == (2, 196, 1152)

def test_pooling_preserves_constant():
    """ test pooling output using constant """

    tokens = torch.ones(2, 729, 1152)
    pooling_output = pool_visual_tokens(tokens)
    assert torch.allclose(pooling_output, torch.ones_like(pooling_output))

def test_pooling_identity():
    """ Verify that pooling to the original grid size preserves every token value and its position. """

    tokens = torch.arange(729, dtype = torch.float32)
    tokens = tokens.reshape(1, 729, 1)
    pooling_output = pool_visual_tokens(tokens, output_size = 27)
    assert torch.equal(tokens, pooling_output)

def test_connector_output_shape():
    """ verify the output shape after pooling and projection """

    tokens = torch.randn(2, 729, 1152)
    pooling_output = pool_visual_tokens(tokens)
    projector = VisualProjector()
    projected_tokens = projector(pooling_output)
    assert projected_tokens.shape == (2, 196, 896)

def test_projector_parameter_count():
    """ verify the total number of parameters in projector """

    projector = VisualProjector()
    param_count = 0

    for parameter in projector.parameters():
        param_count += parameter.numel()

    assert param_count == 1_836_800











