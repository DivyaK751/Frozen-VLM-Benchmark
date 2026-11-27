"""build_model on tiny checkpoints saved to disk (CPU, no downloads). The encoder is saved as a full
SiglipModel (text + vision towers), like the real SigLIP 2 checkpoint, and loaded vision-only.
Run from the repo root: python tests/test_build_model.py"""
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM, SiglipConfig, SiglipModel, SiglipTextConfig, SiglipVisionConfig  # noqa: E402

from _tiny import HIDDEN, IMAGE_PAD, N_IMG, POOL_GRID, TinyDataset, VOCAB, collate  # noqa: E402
from src.model.projector import VisualProjector  # noqa: E402
from src.model.vlm import build_model  # noqa: E402


def test_build_model_from_saved_checkpoints():
    tmp = Path(tempfile.mkdtemp())
    try:
        torch.manual_seed(0)
        vcfg = SiglipVisionConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, image_size=56, patch_size=14)
        tcfg = SiglipTextConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                                num_attention_heads=4, vocab_size=100, max_position_embeddings=16)
        full = SiglipModel(SiglipConfig(text_config=tcfg.to_dict(), vision_config=vcfg.to_dict()))
        full.save_pretrained(tmp / "enc")
        llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
                                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                           max_position_embeddings=256, tie_word_embeddings=True))
        llm.save_pretrained(tmp / "llm")
        proj = VisualProjector(HIDDEN, HIDDEN)
        torch.save(proj.state_dict(), tmp / "projector.pt")

        model = build_model(lora=True, lora_rank=4, lora_alpha=8, lora_dropout=0.0, dtype=torch.float32,
                            pool_grid=POOL_GRID, init_projector=str(tmp / "projector.pt"),
                            encoder_name=str(tmp / "enc"), llm_name=str(tmp / "llm"))
        model.image_pad_id = IMAGE_PAD
        # weights really came from the checkpoint, not from random init
        a = full.vision_model.post_layernorm.weight
        assert torch.equal(model.encoder.vision_model.post_layernorm.weight, a)
        assert torch.equal(model.encoder.vision_model.encoder.layers[0].mlp.fc1.weight,
                           full.vision_model.encoder.layers[0].mlp.fc1.weight)
        assert torch.equal(model.projector.fc1.weight, proj.fc1.weight)            # init_projector loaded
        assert model.num_image_tokens == N_IMG
        b = collate()([TinyDataset(3)[i] for i in range(3)])
        out = model(**b)
        assert torch.isfinite(out.loss)

        # bf16 + gradient checkpointing path
        m2 = build_model(lora=False, dtype=torch.bfloat16, pool_grid=POOL_GRID, gradient_checkpointing=True,
                         encoder_name=str(tmp / "enc"), llm_name=str(tmp / "llm"))
        m2.image_pad_id = IMAGE_PAD
        m2.train()
        assert next(m2.encoder.parameters()).dtype == torch.bfloat16
        assert next(m2.projector.parameters()).dtype == torch.float32
        m2(**b).loss.backward()

        # a checkpoint without vision weights must fail loudly (here: the LLM directory)
        try:
            build_model(encoder_name=str(tmp / "llm"), llm_name=str(tmp / "llm"), pool_grid=POOL_GRID)
            raise AssertionError("expected a failure for missing vision weights")
        except AssertionError:
            raise
        except Exception:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_build_model_from_saved_checkpoints()
    print("ok   test_build_model_from_saved_checkpoints")
    print("ALL CHECKS PASSED")
