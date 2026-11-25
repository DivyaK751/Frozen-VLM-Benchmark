import torch
import torch.nn as nn
from transformers import SiglipVisionModel, Qwen2ForCausalLM, AutoTokenizer

from small_vlm.connector import pool_visual_tokens, VisualProjector 


class SmallVLM(nn.Module):

    def __init__(self, encoder, llm, projector, image_token_id, grid_size = 27, pool_size = 14):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.projector = projector
        self.image_token_id = image_token_id
        self.grid_size=grid_size
        self.pool_size=pool_size
        self.tokens_per_image = self.pool_size ** 2

        input_embeddings = self.llm.get_input_embeddings()
        num_embeddings = input_embeddings.weight.shape[0]
        if self.image_token_id < 0 or self.image_token_id >= num_embeddings:
            raise ValueError(
                f"image_token_id {self.image_token_id} is out of range. " 
                f"Expected 0 through {num_embeddings - 1}."
            )
            

        self.encoder.requires_grad_(False)
        self.llm.requires_grad_(False)

        self.encoder.eval()

    def train(self, mode = True):
        super().train(mode)
        self.encoder.eval()

        return self

    def encode_images(self, pixel_values):

        with torch.no_grad():
            patch_tokens = self.encoder(pixel_values).last_hidden_state

        pooled_tokens = pool_visual_tokens(patch_tokens, grid_size=self.grid_size, output_size=self.pool_size)
        projector_dtype = next(self.projector.parameters()).dtype
        pooled_tokens = pooled_tokens.to(dtype = projector_dtype)
        projected_tokens = self.projector(pooled_tokens)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        return projected_tokens.to(dtype = llm_dtype)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        if input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError(
                f"Expected one image per prompt, got {input_ids.shape[0]} prompts "
                f"and {pixel_values.shape[0]} images."
            )

        image_mask = input_ids == self.image_token_id
        image_counts = image_mask.sum(dim=1)

        if (image_counts != self.tokens_per_image).any():    
            raise ValueError(
                f"expected count {self.tokens_per_image}, got {image_counts.tolist()}"
            )

        image_tokens = self.encode_images(pixel_values)
        embedding_layer = self.llm.get_input_embeddings()
        input_embeds = embedding_layer(input_ids)

        image_tokens = image_tokens.reshape(-1, image_tokens.shape[-1])
        input_embeds[image_mask] = image_tokens

        return self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

def build_small_vlm(*, pool_size=14, dtype=torch.float32):

    encoder, loading_info = SiglipVisionModel.from_pretrained(
        "google/siglip2-so400m-patch14-384",
        dtype=dtype,
        output_loading_info=True,
    )
    missing_keys = loading_info["missing_keys"]
    if missing_keys:
        raise RuntimeError(f"Missing SigLIP vision weights: {sorted(missing_keys)}")

    llm = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError("Qwen tokenizer does not recognize <|image_pad|>.")

    projector = VisualProjector(
        in_dim=encoder.config.hidden_size,
        out_dim=llm.config.hidden_size,
    ).to(dtype=torch.float32)        
    grid_size = encoder.config.image_size // encoder.config.patch_size

    model = SmallVLM(
        encoder=encoder,
        llm=llm,
        projector=projector,
        image_token_id=image_token_id,
        grid_size=grid_size,
        pool_size=pool_size,
    )
    
    return model, tokenizer
    



