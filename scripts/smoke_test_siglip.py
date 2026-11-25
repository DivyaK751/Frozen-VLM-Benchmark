import torch
from transformers import AutoImageProcessor, SiglipModel
from PIL import Image

from small_vlm.model import build_small_vlm

checkpoint = "google/siglip2-so400m-patch14-384"
processor = AutoImageProcessor.from_pretrained(checkpoint)

print("Image mean:", processor.image_mean)
print("Image std:", processor.image_std)
print("Resize size:", processor.size)
print("Resize enabled:", processor.do_resize)
print("Center crop enabled:", getattr(processor, "do_center_crop", "not exposed"))

black_img = Image.new("RGB", (640, 320))
black_pixels = processor(black_img, return_tensors="pt").pixel_values

print("Non-square input:", black_img.size)
print("Processed shape:", black_pixels.shape)
print("Black pixel min/max:", black_pixels.min().item(), black_pixels.max().item())

marked_img = black_img.copy()
marked_img.paste("white", (240, 80, 400, 240))  # 160 × 160 square
marked_pixels = processor(marked_img, return_tensors="pt").pixel_values

rows, cols = (marked_pixels[0, 0] > 0).nonzero(as_tuple=True)
width = (cols.max() - cols.min() + 1).item()
height = (rows.max() - rows.min() + 1).item()
print("Processed marker width and height:", width, height)

model, tokenizer = build_small_vlm()
encoder = model.encoder

print("SigLIP 2 image processor and vision encoder loaded.")

rng = torch.Generator().manual_seed(0)
pixels = torch.randint(0, 256, (384, 384, 3), dtype=torch.uint8, generator=rng)
img = Image.fromarray(pixels.numpy())
pixel_values = processor(img, return_tensors="pt").pixel_values
print(pixel_values.shape)

with torch.no_grad():
    vision_output = encoder(pixel_values)
    print("Patch features:", vision_output.last_hidden_state.shape)
    print("Pooled output:", vision_output.pooler_output.shape)

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_count:,}")
assert trainable_count == 1_836_800, f"Expected 1,836,800, got {trainable_count:,}"

embedding_rows = model.llm.get_input_embeddings().weight.shape[0]
print(f"Tokenizer size: {len(tokenizer):,}")
print(f"Embedding rows: {embedding_rows:,}")
print(f"<|image_pad|> token ID: {model.image_token_id}")
assert tokenizer.encode("<|image_pad|>", add_special_tokens=False) == [
    model.image_token_id
]

image_ids = [model.image_token_id] * model.tokens_per_image
text_ids = tokenizer.encode("This is a black image.", add_special_tokens=False)
input_ids = torch.tensor([image_ids + text_ids], dtype=torch.long)
labels = torch.tensor([[-100] * len(image_ids) + text_ids], dtype=torch.long)

model.eval()
with torch.no_grad():
    outputs = model(
        input_ids,
        pixel_values,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )

assert outputs.loss is not None
assert torch.isfinite(outputs.loss)
print(f"Pretrained smoke-test loss: {outputs.loss.item():.4f}")

print("Qwen tie_word_embeddings:", model.llm.config.tie_word_embeddings)

# Release the assembled model before loading the full SigLIP model.
del encoder, model

full_siglip = SiglipModel.from_pretrained(checkpoint, dtype=torch.float32)
full_siglip.eval()

with torch.no_grad():
    reference = full_siglip.vision_model(pixel_values)

torch.testing.assert_close(
    vision_output.last_hidden_state, reference.last_hidden_state
)
torch.testing.assert_close(
    vision_output.pooler_output, reference.pooler_output
)
print("SigLIP patch features and pooled output match.")