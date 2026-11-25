
import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from small_vlm.model import SmallVLM
from small_vlm.connector import VisualProjector

from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)


class FakeLLM(nn.Module):
    def __init__(self, num_embeddings=1, embedding_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        self.received_attention_mask = attention_mask
        self.received_labels = labels
        return inputs_embeds


def test_rejects_image_token_id_equal_to_embedding_count():
    with pytest.raises(ValueError, match="out of range"):
        SmallVLM(nn.Identity(), FakeLLM(), nn.Identity(), 1)

def test_rejects_negative_image_token_id():
    with pytest.raises(ValueError, match="out of range"):
        SmallVLM(nn.Identity(), FakeLLM(), nn.Identity(), -1)

def test_train_mode_encoder_eval_llm_train():
    encoder = nn.Identity()
    llm = FakeLLM()
    llm.eval()
    model = SmallVLM(encoder, llm, nn.Identity(), 0)
    model.train()

    assert not model.encoder.training
    assert model.llm.training 

class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_tokens = torch.arange(16*8, dtype=torch.float32).reshape(1, 16, 8) * self.scale
        patch_tokens= patch_tokens.expand(batch_size, -1, -1)
        image_offsets = pixel_values[:, 0, 0, 0]
        patch_tokens = patch_tokens + image_offsets.reshape(batch_size, 1, 1)
        self.grad_enabled_during_forward = torch.is_grad_enabled()

        return SimpleNamespace(last_hidden_state = patch_tokens)

def test_encode_images_output_shape():
    encoder = FakeEncoder()
    llm = FakeLLM()
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    pixel_values = torch.zeros(2, 3, 8, 8)
    image_tokens = model.encode_images(pixel_values)

    assert not encoder.grad_enabled_during_forward
    assert image_tokens.shape == (2, 4, 4)
    assert image_tokens.requires_grad

def test_forward_replaces_image_placeholders():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]])
    pixel_values = torch.zeros(2, 3, 8, 8)
    pixel_values[1] = 1
    output = model(input_ids, pixel_values)
    expected_text = llm.embedding.weight[1]

    assert torch.allclose(output[0, 0], expected_text)
    assert torch.allclose(output[0, -1], expected_text)

    expected_image_tokens = model.encode_images(pixel_values)

    assert torch.allclose(output[0, 1:5], expected_image_tokens[0])
    assert torch.allclose(output[1, :4], expected_image_tokens[1])
    assert torch.allclose(output[1, 4], expected_text)
    assert torch.allclose(output[1, 5], expected_text)

def test_forward_rejects_wrong_image_placeholder_count():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[1, 0, 0, 0, 1]])
    pixel_values = torch.zeros(1, 3, 8, 8)

    with pytest.raises(ValueError, match="expected count 4"):
        model(input_ids, pixel_values)

def test_forward_passes_attention_mask_and_labels():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[1, 0, 0, 0, 0, 1, 1]])
    pixel_values = torch.zeros(1, 3, 8, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 1, -100]])
    model(input_ids, pixel_values, attention_mask=attention_mask, labels=labels)

    assert torch.equal(llm.received_attention_mask, attention_mask)
    assert torch.equal(llm.received_labels, labels)

def test_forward_backward_only_projector_gets_gradients():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[1, 0, 0, 0, 0, 1, 1]])
    pixel_values = torch.zeros(1, 3, 8, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 1, -100]])

    output = model(input_ids, pixel_values, attention_mask=attention_mask, labels=labels)
    loss = (output.square()).mean()
    loss.backward()

    assert all(param.grad is not None for param in projector.parameters())
    assert not encoder.scale.requires_grad
    assert encoder.scale.grad is None
    assert all(not param.requires_grad for param in llm.parameters())
    assert all(param.grad is None for param in llm.parameters())

def test_forward_rejects_uneven_placeholder_counts():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]])
    pixel_values = torch.zeros(2, 3, 8, 8)

    with pytest.raises(ValueError, match="expected count 4"):
        model(input_ids,pixel_values)

    assert not hasattr(encoder, "grad_enabled_during_forward")

def test_only_projector_parameters_are_trainable():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)

    assert trainable_count == (8 * 4 + 4) + (4 * 4 + 4)

def test_forward_rejects_mismatched_batch_sizes():
    encoder = FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    input_ids = torch.tensor([[1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1]])
    pixel_values = torch.zeros(1, 3, 8, 8)

    with pytest.raises(ValueError, match="Expected one image per prompt"):
        model(input_ids, pixel_values)

def test_encode_images_preserves_dtype_flow():
    encoder = BF16FakeEncoder()
    llm = FakeLLM(num_embeddings=2, embedding_dim=4).to(dtype=torch.bfloat16)
    projector = VisualProjector(in_dim=8, out_dim=4)
    model = SmallVLM(encoder, llm, projector, image_token_id=0, grid_size=4, pool_size=2)
    pixel_values = torch.zeros(1, 3, 8, 8)

    def check_projector_input_dtype(module, args):
        assert args[0].dtype == torch.float32

    projector.register_forward_pre_hook(check_projector_input_dtype)
    image_tokens = model.encode_images(pixel_values)

    assert image_tokens.dtype == torch.bfloat16

class BF16FakeEncoder(FakeEncoder):

    def forward(self, pixel_values):
        output = super().forward(pixel_values)
        output.last_hidden_state = output.last_hidden_state.to(dtype=torch.bfloat16)
        return output

def test_tiny_siglip_qwen_forward_backward():
    torch.manual_seed(0)
    vision_config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, 
                            num_attention_heads=2, image_size=56, patch_size=14)
    encoder = SiglipVisionModel(vision_config)
    llm_config = Qwen2Config(vocab_size=32, hidden_size=24, intermediate_size=48,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             tie_word_embeddings=True, pad_token_id=0, use_cache=False)
    llm = Qwen2ForCausalLM(llm_config)  
    projector = VisualProjector(in_dim=vision_config.hidden_size, out_dim=llm_config.hidden_size)
    model = SmallVLM(encoder, llm, projector, image_token_id=1, grid_size=4, pool_size=2)

    pixel_values = torch.randn(2, 3, 56, 56)
    input_ids = torch.tensor([[2, 1, 1, 1, 1, 3, 4], [2, 1, 1, 1, 1, 5, 0]],
                             dtype=torch.long)
    attention_mask = (input_ids != llm_config.pad_token_id).to(dtype=torch.long)
    labels = torch.tensor([
        [-100, -100, -100, -100, -100, 3, 4],
        [-100, -100, -100, -100, -100, 5, -100]
        ])
    outputs = model(input_ids, pixel_values, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)

    outputs.loss.backward()

    projector_params = list(projector.parameters())
    assert all(param.grad is not None for param in projector_params)
    assert any((param.grad != 0).any().item() for param in projector_params)

    assert all(param.grad is None for param in encoder.parameters())
    assert all(param.grad is None for param in llm.parameters())

    shared_weight = llm.get_input_embeddings().weight
    assert shared_weight is llm.get_output_embeddings().weight
    assert not shared_weight.requires_grad

    in_dim = vision_config.hidden_size
    out_dim = llm_config.hidden_size
    expected_count = (in_dim * out_dim + out_dim) + (out_dim * out_dim + out_dim)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == expected_count

    




     
       

    

