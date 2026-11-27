"""SmallVLM: frozen SigLIP encoder -> pooling -> trained MLP projector -> Qwen (optionally with LoRA).

Follows DATA_PIPELINE.md section 3. The text already holds NUM_IMAGE_TOKENS placeholder tokens
(<|image_pad|>) per image; this model overwrites their embeddings with the projector output:

    feats  = encoder(pixel_values).last_hidden_state     # [B, 729, 1152], frozen, no grad
    tokens = pool_patches(feats)                         # [B, 169, 1152]
    img    = projector(tokens)                           # [B, 169, 896]
    embeds = embed(input_ids); embeds[input_ids == IMAGE_PAD_ID] = img

Text-only mode (E0 and the "tokens removed" blank test): call forward without pixel_values, with
input_ids that contain no image tokens (get_dataloader(..., text_only=True)).

Dtypes: the encoder and LLM can be bf16 on GPU while the projector stays fp32 (it is the trained
module, so it keeps fp32 master weights). The projector output is cast to the LLM's embedding
dtype before it is spliced in. PEFT keeps the LoRA weights in fp32 too.

E2 and E3 differ only in whether add_lora() is called.
"""

import torch
import torch.nn as nn

from src.config import IM_END_ID, IMAGE_PAD_ID, LLM_NAME, PAD_TOKEN_ID, POOLED_GRID
from src.data.images import ENCODER_NAME
from src.model.pooling import pool_patches
from src.model.projector import VisualProjector

# All 7 linear layers in each Qwen2 block: attention (q, k, v, o) and MLP (gate, up, down).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SmallVLM(nn.Module):
    def __init__(self, encoder, llm, projector, pool_grid=POOLED_GRID, image_pad_id=IMAGE_PAD_ID):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.projector = projector
        self.pool_grid = pool_grid
        self.image_pad_id = image_pad_id
        self.num_image_tokens = pool_grid ** 2        # 169 for the main configuration
        self.has_lora = False

        self.encoder.requires_grad_(False)            # frozen SigLIP
        self.encoder.eval()
        self.llm.requires_grad_(False)                # frozen Qwen (LoRA adds its own trainable weights)

    # ---- modes -----------------------------------------------------------------------------

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()                           # the encoder never leaves eval mode
        return self

    def enable_gradient_checkpointing(self):
        """Trade compute for memory in the LLM. Non-reentrant, so it also works when the
        embeddings are frozen (E0) and with PEFT."""
        self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def add_lora(self, rank=16, alpha=32, dropout=0.05, target_modules=None):
        """Wrap the LLM with LoRA adapters (E3, E0, ablations). Base weights stay frozen.
        With B initialised to zero, the output is unchanged until training starts."""
        from peft import LoraConfig, get_peft_model

        if self.has_lora:
            raise RuntimeError("LoRA already added")
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                            target_modules=target_modules or LORA_TARGET_MODULES,
                            bias="none", task_type="CAUSAL_LM")
        self.llm = get_peft_model(self.llm, config)
        self.has_lora = True
        return self

    # ---- parameters ------------------------------------------------------------------------

    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def count_parameters(self):
        """Returns (trainable, total) parameter counts."""
        trainable = sum(p.numel() for _, p in self.trainable_parameters())
        return trainable, sum(p.numel() for p in self.parameters())

    # ---- forward ---------------------------------------------------------------------------

    def encode_images(self, pixel_values):
        """[B, 3, 384, 384] -> [B, num_image_tokens, llm_hidden], in the LLM embedding dtype."""
        enc_dtype = next(self.encoder.parameters()).dtype
        with torch.no_grad():                                          # encoder is frozen
            feats = self.encoder(pixel_values=pixel_values.to(enc_dtype)).last_hidden_state
        tokens = pool_patches(feats, self.pool_grid)                   # [B, N, enc_hidden]
        proj_dtype = next(self.projector.parameters()).dtype
        img = self.projector(tokens.to(proj_dtype))                    # [B, N, llm_hidden]
        return img.to(self.llm.get_input_embeddings().weight.dtype)

    def build_inputs_embeds(self, input_ids, pixel_values=None):
        """Embed the text and overwrite every <|image_pad|> position with the projected image
        features, in order. Exactly num_image_tokens placeholders per row when images are given,
        and none when they are not (text-only)."""
        embeds = self.llm.get_input_embeddings()(input_ids)
        mask = input_ids == self.image_pad_id
        counts = mask.sum(dim=1)

        if pixel_values is None:
            if counts.any():
                raise ValueError("input_ids contain image tokens but no pixel_values were given")
            return embeds

        if pixel_values.shape[0] != input_ids.shape[0]:
            raise ValueError(f"{input_ids.shape[0]} prompts but {pixel_values.shape[0]} images")
        if (counts != self.num_image_tokens).any():                    # check before running SigLIP
            raise ValueError(f"expected {self.num_image_tokens} image tokens per row, got {counts.tolist()}")

        img = self.encode_images(pixel_values)                         # [B, N, D]
        embeds = embeds.clone()
        embeds[mask] = img.reshape(-1, img.shape[-1])                  # row-major, like the placeholders
        return embeds

    def forward(self, input_ids, attention_mask=None, labels=None, pixel_values=None, **kwargs):
        """Returns the Hugging Face causal-LM output (.loss, .logits). Extra kwargs such as
        num_items_in_batch go to the loss, which the trainer uses for exact gradient accumulation."""
        embeds = self.build_inputs_embeds(input_ids, pixel_values)
        return self.llm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values=None, max_new_tokens=32,
                 eos_token_id=IM_END_ID, pad_token_id=PAD_TOKEN_ID):
        """Greedy decoding for evaluation. Prompts must be LEFT-padded (the training collator pads on
        the right). Returns only the new tokens, [B, <=max_new_tokens].

        Qwen's default generation config sets a repetition penalty and sampling parameters; they are
        switched off here so the decoding is plain greedy, the same for every model."""
        embeds = self.build_inputs_embeds(input_ids, pixel_values)
        return self.llm.generate(inputs_embeds=embeds, attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                 eos_token_id=eos_token_id, pad_token_id=pad_token_id)

    def merge_lora(self, adapter_dir):
        """Load a trained LoRA adapter (final/lora from training) and merge it into the LLM weights,
        leaving a plain model that is as fast as the base one."""
        from peft import PeftModel

        if self.has_lora:
            raise RuntimeError("this model already has trainable LoRA adapters")
        self.llm = PeftModel.from_pretrained(self.llm, str(adapter_dir)).merge_and_unload()
        return self


def _dtype_kwargs(dtype):
    """transformers 4.56+ calls the argument `dtype`; older 4.x versions call it `torch_dtype`."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def build_model(*, lora=False, lora_rank=16, lora_alpha=32, lora_dropout=0.05, dtype=torch.bfloat16,
                pool_grid=POOLED_GRID, gradient_checkpointing=False, init_projector=None,
                encoder_name=ENCODER_NAME, llm_name=LLM_NAME, freeze_projector=False):
    """Load the real checkpoints and assemble the model.

    init_projector   path to a projector state dict (the Stage 1 result) to start Stage 2 from
    freeze_projector for text-only runs (E0), where the projector is never used
    """
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    # SigLIP 2 So400m at 384 px is a fixed-resolution checkpoint that loads with the plain Siglip
    # vision classes. Fail loudly if any vision weight is missing instead of running with random ones.
    encoder, info = SiglipVisionModel.from_pretrained(encoder_name, output_loading_info=True, **_dtype_kwargs(dtype))
    if info["missing_keys"]:
        raise RuntimeError(f"missing SigLIP vision weights: {sorted(info['missing_keys'])[:5]} ...")
    llm = AutoModelForCausalLM.from_pretrained(llm_name, **_dtype_kwargs(dtype))

    projector = VisualProjector(encoder.config.hidden_size, llm.config.hidden_size).float()
    if init_projector:
        projector.load_state_dict(torch.load(init_projector, map_location="cpu"))

    model = SmallVLM(encoder, llm, projector, pool_grid=pool_grid)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if lora:
        model.add_lora(lora_rank, lora_alpha, lora_dropout)
    if freeze_projector:
        model.projector.requires_grad_(False)
    return model
