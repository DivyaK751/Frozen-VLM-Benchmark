# Building a Small Vision-Language Model: How Far Can a Frozen Encoder and a Trained Connector Go?

*Revised project plan, updated after instructor feedback*

## 1. Title, team and members

**Team 14** · Idea #13 (Building a small vision-language model) · Foundation Model Engineering, Fall 2026

**Members:** Divyasri Kadekar (dk3480) · Lakshya Khandelwal (lk3068) · Ritwik Sharan (rs4764) · Vinod Walia (vw2317)

## 2. Summary

Models that answer questions about images, such as GPT-4V, are large and expensive. We ask how much of that ability a small, cheap model can reach. We follow the LLaVA recipe and connect three parts:

- a frozen, pretrained image encoder (SigLIP 2 So400m) that turns a picture into features;
- a small trained network (a two-layer MLP "projector") that maps those features into the language model's input space;
- a small language model (Qwen2.5-0.5B) that writes the answer.

We first train on 558K image–caption pairs so that the projector links images to words. We then train on 150K instruction-following conversations. Our main experiment compares two ways of doing the second stage: updating only the projector, or also adapting the language model with LoRA, a low-cost fine-tuning method. Each condition runs with two seeds, and we report bootstrap confidence intervals.

We measure three things:

- science-diagram reasoning (ScienceQA);
- reading text inside images (TextVQA);
- object hallucination, meaning claims about objects that are not in the picture (POPE).

Two controls check whether answers actually use the image. The first is a precisely defined blank-image test. The second is a text-only model fine-tuned on the same conversations. We also run a LoRA-rank ablation.

Our training data was not designed for OCR or multiple-choice science questions. We therefore report format-failure rates and treat near-floor scores as valid results. We expect LoRA to beat projector-only training and to reduce hallucination, and we expect the projector-only model to rely more on language priors. The outcome is a set of measured trade-offs, at an estimated $239–354 of our $400 in credits, showing which design choices matter for small VLMs.

## 3. Motivation and impact

Small vision-language models (VLMs) run on a laptop or a single cheap GPU, which allows private, offline and low-budget use. Two open questions hold back their adoption:

- How much quality is lost when the encoder is frozen and only a small connector is trained?
- How often do these models describe things that are not there?

Hallucination makes VLMs unsafe for uses such as describing scenes for visually impaired users. Our results will help engineers choose a cheap recipe (what to train and what to freeze), and they will show deployers where and how these models hallucinate.

## 4. Related work

- **LLaVA** (Liu et al., 2023) introduced visual instruction tuning with 7–13B LLMs. We study the same recipe at 0.5B.
- **LLaVA-1.5** (Liu et al., 2024) added an MLP projector, full LLM fine-tuning and academic-task data with short-answer format prompts. We compare projector-only training with LoRA on a small budget, without the academic-task mix.
- **TinyLLaVA** (Zhou et al., 2024) surveys small LLM/encoder pairs. We instead run one controlled ablation with multiple seeds, plus hallucination and grounding analysis.
- **SmolVLM** (Marafioti et al., 2025) pairs a SigLIP So400m encoder with visual-token compression for small models. We adopt this encoder with average pooling of its patch tokens and test what the connector and LoRA each contribute.
- **POPE** (Li et al., 2023) measures object hallucination. We use it and break down failures by split and object type.

## 5. Research question and hypotheses

**Main question:** With a frozen SigLIP 2 So400m encoder (169 visual tokens after pooling) and a 0.5B Qwen2.5 language model, how much does adding LoRA to the language model improve ScienceQA-IMG accuracy, TextVQA accuracy and POPE F1, compared with training only the MLP projector? On which object types does each model hallucinate?

**Fixed definitions (set before any run).** These definitions are fixed before we see any results.

- **Gain.** For a benchmark *b*, Δ*b* = score(E3) − score(E2), in absolute percentage points. Each score is the mean over two seeds.
- **Significant gain.** The 95% paired-bootstrap CI of Δ*b* excludes 0. We resample test items 1,000 times and recompute both models' scores on each resample.
- **Largest gain.** Δ*b* for one benchmark is larger than Δ*b* for each other benchmark, and the bootstrap CI of the difference between the two gains excludes 0. If a CI includes 0, we report the result as "no clear largest gain."

**Hypotheses.**

- **H1.** Projector + LoRA (E3) shows a significant gain over projector-only (E2) on all three benchmarks, and the largest gain is on ScienceQA. Because ScienceQA gains may come from better answer formatting, we also report Δ on the subset of answers that parse correctly. H1 holds for "reasoning" only if the gain survives on that subset.
- **H2.** E2 hallucinates more than E3: it has lower POPE F1 and a higher "yes" rate. The F1 gap (E3 − E2) is largest on POPE's adversarial split (objects that often co-occur with those in the image), compared with the random and popular splits.
- **H3.** Blanking the image lowers TextVQA and POPE much more than ScienceQA, measured as chance-normalized relative drop (Section 7). The drop is smaller for E2 than for E3, which would show that E2 relies more on language priors. Both models also score above the text-only model tuned on the same data (E0).

## 6. Approach

### Architecture

The pipeline runs in this order:

1. The input image is resized to 384 px.
2. The frozen SigLIP 2 So400m encoder (patch size 14) turns it into **729 patch tokens** (a 27×27 grid).
3. Adaptive average pooling to 13×13 reduces this to **169 visual tokens**. Because 27 is odd, plain 2×2 pooling would drop the last row and column (53 of the 729 patches, a strip along the right and bottom edges). Adaptive pooling instead averages overlapping 3×3 windows with stride 2 that cover the whole grid, so every patch contributes.
4. A 2-layer MLP projector with GELU maps the features from 1152 to 896 dimensions. This projector is the trained part.
5. Qwen2.5-0.5B-Instruct reads the 169 visual tokens, then the text prompt, and generates the answer.

**Visual-token configurations.** The main configuration uses **169 tokens** (729 before pooling). The stretch variants bracket it: **81 tokens** (9×9, from 3×3 pooling with stride 3) is more compressed, and **729 tokens** (no pooling) is the full encoder output. All three use every patch, so they differ only in how strongly the image is compressed.

### Training

**Stage 1, alignment (E1).** The encoder and the LLM are frozen. We train only the projector on LLaVA-Pretrain for one epoch, using one seed. All Stage 2 runs start from this checkpoint.

**Stage 2, instruction tuning.** We train on LLaVA-Instruct-150K, which we checked has no images in common with POPE (see below). All conditions use the same data, the same number of steps and the same hyperparameters, with seeds {0, 1}:

| Run | Trained parameters | Seeds | Purpose |
|---|---|---|---|
| **E2** | Projector only | 0, 1 | Main comparison |
| **E3** | Projector + LoRA (rank 16, α = 32) on all attention and MLP projections of the LLM | 0, 1 | Main comparison |
| **E0** | LoRA (rank 16) on the LLM only, trained on the text of the same conversations with all image tokens removed | 0 | Text-only control tuned on the same data |
| **A1 / A2** | Same as E3, with LoRA rank 4 / rank 64 | 0 | Our own ablation: LoRA rank |

Seed variance covers Stage 2 only. Stage 1 is shared by all runs, and we state this limitation in the report.

**Our own ablation: LoRA rank.** We chose LoRA rank (4 / 16 / 64) as our ablation because it needs only Stage 2 reruns. It also tests directly whether the gain from E2 to E3 comes from LLM capacity. We compare the A1 and A2 results with the seed range of E3: a difference that falls inside the seed-to-seed spread is reported as no effect. Token count (81 / 729) remains a stretch goal, because each variant needs a new Stage 1 run.

### Train/test overlap with POPE

POPE uses 500 COCO **val2014** images. LLaVA-Instruct-150K uses COCO images distributed under their train2017 file names, and train2017 contains most of val2014, so overlap was possible. We checked as follows:

- We matched COCO image IDs between the two sets (`scripts/check_pope_overlap.py`). **None of the 500 POPE images appear in LLaVA-Instruct-150K** (0 of 81,479 unique training images; 0 of 157,712 conversations), so no Stage 2 data needed to be removed. This is consistent with LLaVA-Instruct being built from COCO train2014 images, which never overlap with val2014. The script removes matching conversations automatically if the data changes.
- LLaVA-Pretrain uses LAION/CC/SBU images rather than COCO. We run a perceptual-hash near-duplicate check against the 500 POPE images and report what it finds.
- TextVQA (OpenImages) and ScienceQA images do not come from COCO. We cannot control what SigLIP 2 or Qwen2.5 saw during their own pretraining, and we state this as a limitation.

### Methods and tools

- **Course methods:** transfer learning from frozen foundation models, LoRA, instruction tuning, bf16 mixed precision with gradient checkpointing, and ablations.
- **Tools:** built on Transformers + PEFT; evaluation uses lmms-eval with its standard prompts.

| Component | Source | License |
|---|---|---|
| Vision encoder | google/siglip2-so400m-patch14-384 | Apache 2.0 |
| Language model | Qwen/Qwen2.5-0.5B-Instruct | Apache 2.0 |
| Stage 1 data | liuhaotian/LLaVA-Pretrain (558K pairs) | Research use; LAION/CC/SBU terms |
| Stage 2 data | liuhaotian/LLaVA-Instruct-150K (COCO images) | CC BY 4.0; OpenAI terms |
| Evaluation | ScienceQA, TextVQA, POPE | CC BY-NC-SA 4.0; CC BY 4.0; MIT |

## 7. Evaluation plan

### Metrics

- **ScienceQA image subset (test):** multiple-choice accuracy.
- **TextVQA (val):** VQA accuracy.
- **POPE (random, popular and adversarial splits):** F1, accuracy and "yes" rate, broken down by COCO object category and supercategory.
- **Uncertainty:** every headline number is reported as the mean over two seeds with a 95% bootstrap CI (1,000 resamples over test items). For E2 vs. E3, the CI is on the paired difference.

### Format failures and near-floor results

LLaVA-Instruct-150K contains no OCR data and no multiple-choice science data. TextVQA may therefore score near the floor, and ScienceQA scores may mostly reflect whether the model outputs a valid option letter. For every model and benchmark we report a **format-failure rate**, defined as follows:

- **ScienceQA:** the output does not parse to a valid option letter.
- **POPE:** the output does not begin with "yes" or "no".
- **TextVQA:** the output is empty or longer than 10 words.

Format failures count as wrong in the main score. We also report accuracy on the answers that do parse. **A near-floor TextVQA score is a valid result:** it would show that this data recipe does not teach reading text in images. We will not add OCR data after seeing results.

### Baselines and controls

**Text-only LLM, zero-shot.** The base Qwen2.5-0.5B-Instruct, given no image. It shows how much each benchmark can be answered from text alone.

**Text-only LLM, tuned on the same data (E0).** This model separates "LoRA adapted the LLM to the answer style" from "LoRA helped the model use the image."

**Blank-image tests.** These run on seed 0 of E2 and E3:

- **Main test, gray image.** Each image is replaced with a uniform 384×384 image of the SigLIP mean pixel value and passed through the full pipeline. The model therefore still receives 169 visual tokens in the same positions.
- **Secondary test, tokens removed.** The visual tokens are dropped and the model sees only the text prompt. This version checks that the gray image is not itself a misleading input.
- **Relative drop.** We report the chance-normalized relative drop (S_img − S_blank) / (S_img − S_chance). The chance levels are: ScienceQA, the mean of 1/(number of options); TextVQA, 0; POPE accuracy, 50%. We also report absolute drops.

**Other comparisons.**

- Projector-only vs. projector + LoRA, with the same data, seeds and steps.
- The LoRA-rank ablation (ranks 4, 16 and 64).
- Published LLaVA-1.5-7B and TinyLLaVA scores, which we do not re-run. These were trained on mixtures that include VQA and OCR data, so the comparison is not like-for-like.
- POPE chance level (50%).

### Why these measures fit

They test the three skills we target: reasoning, fine visual detail and hallucination. The text-only, tuned text-only and blank-image controls separate real image use from language priors and answer-format learning.

### What they will not capture

- POPE checks only yes/no object existence. It misses hallucinated attributes, counts and relations, as well as errors in long free-form descriptions.
- ScienceQA can be partly solved from text priors.
- Two seeds give a rough estimate of variance, not a precise one.

To cover part of this gap, we hand-inspect about 50 generated descriptions per model for E2 and E3.

## 8. Compute and cost plan

All training and evaluation run on Google Cloud Vertex AI custom training, paid from pooled team credits of $400 ($100 × 4). We use one A100 40 GB (a2-highgpu-1g) at the on-demand Vertex rate of about $4.43 per hour, including the management fee.

| Experiment | GPU-h | USD |
|---|---|---|
| E1: Stage 1 alignment (558K pairs, 1 seed) | 7–10 | 31–44 |
| E2 + E3: Stage 2, 2 seeds each (4 runs) | 20–30 | 89–133 |
| E0: text-only LoRA on the same conversations | 2–3 | 9–13 |
| Ablation: LoRA rank 4 and 64 (seed 0) | 10–15 | 44–66 |
| Evaluation (12 runs; see below) | 10–14 | 44–62 |
| Pilots and reruns | 5–8 | 22–35 |
| **Total** | **54–80** | **239–354** |

**The 12 evaluation runs:**

- zero-shot text-only Qwen;
- E0;
- E2 and E3, seeds 0 and 1 (four runs);
- gray-image and tokens-removed tests for E2 and E3, seed 0 (four runs);
- the two rank-ablation models.

**Spending order and cutoffs.** Runs go in this order: E1, then E2 and E3 at seed 0, then E2 and E3 at seed 1, then E0, then the rank ablation.

- If we have spent more than $300 before the ablation, we run only rank 64.
- The 81- and 729-token stretch variants run only if at least $100 remains after the core plan. Otherwise they are dropped.

## 9. Timeline and roles

| Dates | Milestone | Done when |
|---|---|---|
| Nov 6–13 | Abstract; setup and pilot | Data ready; 10K pilot runs end to end; zero-shot text-only baseline scored |
| Nov 14–20 | Core experiments, seed 0 | Overlap filtering done and counts recorded; E1, E2 and E3 (seed 0) trained and scored |
| Nov 21–27 | Seeds, controls and ablation | E2 and E3 at seed 1; E0; LoRA rank 4 and 64; all scored with format-failure rates |
| Nov 28–Dec 1 | Analysis | Bootstrap CIs; H1–H3 tested against the fixed definitions; blank-image tests; POPE failures by split and object type; manual check |
| Dec 2–6 | Results and demo | Results tables with CIs, E2 vs. E3 and rank-ablation plots, demo, slides |
| Course date | Presentation and report | Final report against this plan |

**Workstream leads:**

- **Ritwik Sharan – training pipeline:** data download and preprocessing, POPE-overlap filtering, chat formatting, Stage 1/2 training loop, E0 text-only run, Vertex jobs and checkpointing.
- **Divyasri Kadekar – connector:** So400m encoder setup, 13×13 adaptive pooling (169 tokens), MLP projector, E1 alignment, and E2 projector-only runs (both seeds).
- **Vinod Walia – LoRA:** LoRA configuration on Qwen (rank and target layers), E3 projector + LoRA runs (both seeds), LoRA-rank ablation, and training-curve comparison of E2 vs. E3.
- **Lakshya Khandelwal – evaluation:** ScienceQA, TextVQA and POPE with lmms-eval; format-failure parsing; text-only and blank-image baselines; bootstrap CIs; POPE hallucination breakdown; results tables.

## 10. Risks and fallback

**Most likely failure: training is slower or costlier than estimated.** The fallbacks, in order:

1. Drop the rank-4 run.
2. Train Stage 1 on a 200K-caption subset.
3. If still needed, pool to 81 tokens.

The two-seed E2 vs. E3 comparison and the E0 control are protected: they are the last things we would cut. We save checkpoints every 500 steps.

**Other risks:**

- **Silent chat-template or image-token bugs.** We first overfit 100 examples and inspect the inputs.
- **GPU memory limits with So400m at 384 px.** We handle these with gradient checkpointing and smaller batches.
- **Near-floor TextVQA or ScienceQA scores.** We report them, with format-failure rates, as valid results rather than tuning prompts after seeing them.

## 11. References

1. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. *NeurIPS 2023*. arXiv:2304.08485.
2. Liu, H., Li, C., Li, Y., & Lee, Y. J. (2024). Improved Baselines with Visual Instruction Tuning. *CVPR 2024*. arXiv:2310.03744.
3. Zhou, B., Hu, Y., Weng, X., Jia, J., Luo, J., Liu, X., Wu, J., & Huang, L. (2024). TinyLLaVA: A Framework of Small-scale Large Multimodal Models. arXiv:2402.14289.
4. Marafioti, A., et al. (2025). SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299.
5. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., & Wen, J.-R. (2023). Evaluating Object Hallucination in Large Vision-Language Models (POPE). *EMNLP 2023*. arXiv:2305.10355.
6. Tschannen, M., et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786.
7. Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
8. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv:2106.09685.
9. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., & Kalyan, A. (2022). Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (ScienceQA). *NeurIPS 2022*. arXiv:2209.09513.
10. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., & Rohrbach, M. (2019). Towards VQA Models That Can Read (TextVQA). *CVPR 2019*.
11. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*.
12. Zhang, K., et al. (2024). LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models. arXiv:2407.12772.
13. Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. GitHub: huggingface/peft.
14. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

## 12. AI-use disclosure

We used an AI assistant (Claude, Anthropic) to help revise this plan in response to instructor feedback. It restructured the text, drafted the new sections (fixed hypothesis definitions, blank-image protocol, overlap check, cost table and references) and checked the arithmetic. The team reviewed and edited all content. All design decisions, cost estimates and citations were verified by team members, and we take responsibility for them.

During the project we expect to use AI coding assistants for boilerplate code (data loaders, evaluation scripts and plotting). All experiments, results and analysis will be run and checked by the team, and the final report will include an updated disclosure.

---

## Appendix: Response to feedback

| Feedback | Change |
|---|---|
| TextVQA may be near floor; ScienceQA may mostly reflect format | Format-failure rates and accuracy on parseable answers for every model (§7). Near-floor results are stated as valid outcomes, and H1's ScienceQA claim must hold on parseable answers (§5). |
| POPE (COCO val2014) overlaps with COCO training images | COCO image-ID matching found 0 of 500 POPE images in LLaVA-Instruct-150K, so nothing was removed. A pHash check of all 558K LLaVA-Pretrain and 118K COCO training images found no near-duplicates of POPE images (§6). |
| At least 2 seeds; bootstrap CIs; define "largest gain" | E2 and E3 with seeds {0, 1}; 95% paired-bootstrap CIs; gain, significant gain and largest gain defined before any run (§5, §7). |
| Specify the blank-image test; use relative drops; compare with a text-only model tuned on the same data | Main test: mean-pixel gray image; secondary test: tokens removed; chance-normalized relative drop; new E0 text-only LoRA control (§6, §7). |
| State the visual-token count | 729 tokens (27×27) before pooling → 169 (13×13) after adaptive average pooling, which uses every patch; the 81 (9×9) and 729 (no pooling) variants bracket the main configuration (§6). |
| Add one ablation of our own | LoRA rank 4 / 16 / 64, compared with the E3 seed spread (§6, §8). |
| References list and AI-use disclosure | Added (§11, §12). |
