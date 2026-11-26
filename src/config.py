"""Shared format constants (task L2). Every part of the pipeline imports these from here,
so the image-token format is defined in one place. See data_and_lora_plan.md,
"Image-token format".
"""

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Image tokens: <|vision_start|> + NUM_IMAGE_TOKENS x <|image_pad|> + <|vision_end|>.
# All three already exist in the Qwen2.5 tokenizer as single tokens (checked in L1),
# so no tokens are added and the embeddings are not resized.
NUM_IMAGE_TOKENS = 169          # 27x27 SigLIP grid -> 2x2 avg pool -> 13x13
VISION_START_TOKEN = "<|vision_start|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_END_TOKEN = "<|vision_end|>"
VISION_START_ID = 151652
IMAGE_PAD_ID = 151655           # the model swaps these positions for the projector output
VISION_END_ID = 151653

IM_START_ID = 151644            # <|im_start|>
IM_END_ID = 151645              # <|im_end|>, also Qwen's EOS for chat generation
PAD_TOKEN_ID = 151643           # <|endoftext|>, Qwen2.5's pad token

# Qwen2.5's default system prompt (what its chat template inserts when no system
# message is given), so training matches lmms-eval's default prompting.
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Stage 1 replaces LLaVA-Pretrain's varied caption requests with one fixed instruction.
STAGE1_PROMPT = "Describe the image."

MAX_LEN = 2048
IGNORE_INDEX = -100
