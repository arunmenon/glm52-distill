# Student Candidate Menu (GLM-5.2 teacher)

From the deep-research workflow (2026-07-06, 111 agents, adversarially verified).
The menu splits by tokenizer: **only GLM-family models can take logit KD**
(top-20 logprobs need a shared vocab); everything else is sequence-KD only.

## GLM-family — logit-KD capable (Student A path)

| Model | HF repo | Params | Dense/MoE | License | Notes |
|---|---|---|---|---|---|
| **GLM-4.5-Air** | zai-org/GLM-4.5-Air | 106B / 12B active | MoE | MIT | Current Student A. **We verified vs GLM-5.2 → `remap` mode** (36 moved tokens, 3.5k teacher-only, ~151k shared). Strongest GLM student; 12B active = moderate serve cost. |
| **GLM-Z1-32B-0414** | zai-org/GLM-Z1-32B-0414 | 32B | Dense | MIT | NEW candidate. Reasoning-tuned, DENSE (far simpler to serve than a 106B MoE), fits the 70B ceiling cleanly. **Tokenizer compat to GLM-5.2 UNVERIFIED — run 00_check_tokenizer.py before trusting logit KD.** |
| **GLM-Z1-9B-0414** | zai-org/GLM-Z1-9B-0414 | 9B | Dense | MIT | Smaller GLM logit-KD option; same tokenizer caveat. |

## Non-GLM — sequence-KD only (Student B path)

| Model | HF repo | Params | Dense/MoE | License | Notes |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | Qwen/Qwen3-30B-A3B-Instruct-2507 | 30.5B / 3.3B active | MoE | Apache 2.0 | Best DEPLOY economics — 30B quality at 3.3B-active inference. Already cached on an earlier pod. |
| **Qwen3-32B** | Qwen/Qwen3-32B | 32B | Dense | Apache 2.0 | Max Qwen3 quality, dense = simplest serve. |
| **Qwen3-Coder-30B-A3B** | Qwen/Qwen3-Coder-30B-A3B-Instruct | 30.5B / 3.3B active | MoE | Apache 2.0 | Coding-focused variant of the 30B MoE. |
| **Qwen3-14B** | Qwen/Qwen3-14B | 14B | Dense | Apache 2.0 | Cheapest step up from the 8B v0. |
| Qwen3-8B | Qwen/Qwen3-8B | 8B | Dense | Apache 2.0 | Current v0 student. |
| Phi-4 | microsoft/phi-4 | 14B | Dense | MIT | Data-quality-focused alternative (tiktoken vocab). |

Other families (Gemma 3 ~262k SentencePiece, Llama, Mistral, DeepSeek) are all
non-GLM tokenizers → sequence-KD only; Qwen3 dominates the non-GLM menu on
license (Apache, no MAU clause vs Llama's 700M-MAU) + benchmarks + deploy cost.

## Recommendation for Tier 3

- **Logit-KD student: keep GLM-4.5-Air** (remap-verified with GLM-5.2). Optionally
  spend ~$1 to run the tokenizer gate on **GLM-Z1-32B-0414** — if it passes, a
  dense 32B logit-KD student is dramatically cheaper to serve than the 106B Air
  MoE and could become the preferred Student A.
- **Sequence-KD student: add Qwen3-30B-A3B** (deploy economics) or **Qwen3-32B**
  (max quality) as the third student, alongside the existing Qwen3-8B.

Caveat carried from the research: GLM-Z1 tokenizer identity with GLM-5.2 is an
assumption until the gate is run; Air's compatibility is ground-truth (we tested it).
