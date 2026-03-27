# Budget-Aware Sparse Knowledge Distillation for LLMs

## Immediate Priority: Implement the Baselines First

Before implementing our proposed methods, we should first build a clean and stable baseline pipeline with the following three methods:

1. **Full KD (Unconstrained)**
2. **Top-K KD**
3. **Sampling-based KD**  
   - For this baseline, follow the idea from **Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs**
   - Main idea: instead of storing the full teacher distribution or only the Top-K logits, sample a subset of teacher logits from the teacher distribution and use that subset to approximate the teacher signal.

These three baselines are the most important first step because:
- they define the reference points for our project,
- they test our teacher-cache and training pipeline,
- and they give us meaningful comparisons before building heuristic tail summaries or compact tail representations.

---

## Project Goal

The project studies how to represent teacher outputs efficiently during language-model distillation when we cannot afford to store the full teacher distribution over the vocabulary for every token.

The core question is:

> Under a fixed per-token teacher-information budget, how should retained teacher information be represented so that the student preserves as much useful signal as possible?

The project is **not** trying to solve sparse KD in full generality, and it is **not** claiming to discover the globally best sparse KD method. The narrower and safer focus is:

- compare several representative sparse teacher representations under the same fixed per-token budget,
- emphasize preserving useful information **beyond the Top-K portion** of the teacher distribution,
- propose better sparse teacher representations, especially for the **tail**.

---

## Main Experimental Setting

Use **offline, token-level, continued-pretraining-style distillation**.

That means:
- start from a pretrained teacher checkpoint,
- start from a pretrained student checkpoint,
- distill on a moderate text corpus,
- cache teacher outputs offline,
- compare methods under the same per-token budget.

This is the cleanest fit for our question because:
- the supervision target is the next-token distribution,
- the budget is naturally defined per token,
- sparse teacher representations can be cached explicitly and compared fairly,
- and we avoid extra confounders from instruction tuning or downstream tasks.

---

## Recommended Experimental Setup

### Model family
Use the same model family for teacher and student.

Recommended first setup:
- **Teacher:** Pythia-1.4B
- **Student:** Pythia-160M

Possible stronger student alternative:
- **Teacher:** Pythia-1.4B
- **Student:** Pythia-410M

The safest first-pass choice is:
- **Teacher:** Pythia-1.4B
- **Student:** Pythia-160M

### Data
Recommended:
- **Train:** OpenWebText subset
- **Validation/Test:** WikiText-103

### Sequence length
- Start with **256**
- Increase only if everything is stable

### Budget levels
Define a fixed **per-token** teacher-information budget:
- **B = 4**
- **B = 8**
- **B = 16**

Budget means the number of teacher-derived scalar values stored per token.

---

## Baselines to Implement First

### 1. Full KD (Unconstrained)

#### What it is
Store the full teacher distribution over the vocabulary at every token position and train the student using:
- standard next-token cross-entropy loss,
- plus KD loss against the full teacher distribution.

#### Why it matters
- serves as the upper bound,
- verifies our full teacher-student training pipeline,
- gives us a dense reference for all sparse methods.

#### Loss
\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{KD}
\]

where:
- \(\mathcal{L}_{CE}\) = next-token cross-entropy
- \(\mathcal{L}_{KD}\) = KL between full teacher and student distributions

---

### 2. Top-K KD

#### What it is
Store only the Top-K teacher logits for each token.  
Use only those logits to supervise the student.

Typical behavior:
- deterministic sparse approximation,
- efficient and simple,
- but biased because the tail is discarded.

#### Why it matters
- strongest standard sparse baseline,
- directly relevant to our project,
- necessary reference point for all tail-aware methods.

#### Important implementation choice
Use teacher-selected Top-K token ids and logits:
- gather student logits on the same ids,
- renormalize over Top-K support,
- compute KL on restricted support.

#### Loss
\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{head}
\]

where:
- \(\mathcal{L}_{head}\) = KL between teacher and student over the Top-K support

---

### 3. Sampling-based KD

#### What it is
Approximate the teacher distribution by sampling a subset of teacher tokens according to the teacher distribution.

This baseline follows the idea from:
- **Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs**

#### Why it matters
- avoids the deterministic truncation bias of Top-K,
- provides a more principled sparse approximation,
- gives us a strong baseline against which to compare our methods.

#### High-level implementation idea
For each token position:
1. compute teacher distribution,
2. sample a subset of vocabulary items according to teacher probabilities,
3. store sampled ids and corresponding teacher probabilities/logits,
4. train the student using this sampled teacher support.

#### Important note
Sampling-based KD is usually:
- less biased than Top-K in expectation,
- but noisier because of sampling variance.

#### Loss
Use the sampled support to compute a sparse KD loss between teacher and student on the sampled indices.

---

## Proposed Methods (After Baselines)

### Approach 1: Heuristic Tail Summaries
Represent retained teacher information as:
- Top-K head supervision
- plus a small number of heuristic tail summary statistics

Examples:
- tail mass
- tail entropy
- mass + entropy

Goal:
- test whether simple handcrafted summaries can preserve useful tail information under a strict budget

### Approach 2: Compact Tail Representation
Represent retained teacher information as:
- Top-K head supervision
- plus a compact learned representation of the tail

Recommended order:
- first implement **PCA**
- then, only if time allows, implement **autoencoder**

Goal:
- test whether compact learned representations preserve more useful signal than simple summaries

---

## Budget Definition

We use a **per-token** budget.

Let \(B\) be the number of teacher-derived scalar values stored per token position.

This budget may include:
- Top-K logits,
- sampled logits,
- tail summary statistics,
- compact latent codes.

The key principle is:
- all methods should be compared under the same fixed per-token budget.

We do **not** use a global dataset-level budget, because that would mix:
- representation quality
- and coverage of supervised tokens

and make comparisons harder to interpret.

---

## Recommended Implementation Order

### Phase 0: Lock the setup
Before heavy implementation, fix:
- teacher checkpoint,
- student checkpoint,
- dataset,
- sequence length,
- budgets,
- exact baseline methods.

Recommended initial config:
- teacher = Pythia-1.4B
- student = Pythia-160M
- train = OpenWebText subset
- eval = WikiText-103
- seq_len = 256
- budgets = [4, 8, 16]

---

### Phase 1: Build Full KD pipeline first

#### Tasks
1. Load teacher model
2. Load student model
3. Tokenize corpus
4. Run teacher forward pass
5. Run student forward pass
6. Compute CE loss
7. Compute full KD loss
8. Train for a few steps

#### Milestone
- one batch runs end-to-end
- loss is finite
- student can overfit a tiny subset

---

### Phase 2: Build teacher cache format

#### Goal
Define a reusable cache format for all methods.

#### Suggested contents per token
- token position reference
- teacher Top-K ids
- teacher Top-K logits
- optional sampled ids / values
- optional tail statistics
- optional PCA / latent codes

#### Important
Design the cache so multiple methods can reuse the same base teacher run.

#### Milestone
- can generate cached teacher outputs for a small dataset split

---

### Phase 3: Implement Top-K KD

#### Tasks
1. Extract teacher Top-K ids/logits
2. Gather student logits on teacher-selected ids
3. Renormalize over restricted support
4. Compute Top-K KD loss
5. Train with CE + head KD

#### Milestone
- Full KD and Top-K KD both run on the same train split
- validation PPL can be computed for both

---

### Phase 4: Implement Sampling-based KD

#### Tasks
1. Sample token ids from teacher distribution
2. Store sampled support per token
3. Gather corresponding student logits
4. Compute sparse KD loss on sampled support
5. Compare against Full KD and Top-K

#### Milestone
- Full KD / Top-K / Sampling all work end-to-end

---

### Phase 5: Implement Heuristic Tail Summaries

#### Tasks
1. Compute teacher tail mass
2. Compute teacher tail entropy
3. Compute corresponding student-side quantities
4. Define summary-matching loss
5. Run variants:
   - mass only
   - entropy only
   - mass + entropy

#### Milestone
- can compare:
  - Top-K
  - Top-K + mass
  - Top-K + entropy
  - Top-K + mass + entropy

---

### Phase 6: Implement Compact Tail Representation

#### Recommended order
Start with **PCA**, not autoencoder.

#### Tasks
1. Define tail representation input
   - recommended: sorted tail probabilities
2. Fit PCA offline on teacher tail vectors
3. Store PCA code per token
4. Compute student tail code
5. Match teacher vs student code

#### Optional later
- add autoencoder-based tail compression

#### Milestone
- PCA tail representation works end-to-end

---

### Phase 7: Build evaluation scripts

#### Quantitative metrics
- NLL
- PPL
- storage per token
- total cache size

#### Main quantitative comparisons
- performance vs budget
- Full KD vs Top-K vs Sampling vs our methods

#### Milestone
- one evaluation script works for all saved checkpoints

---

### Phase 8: Build qualitative analysis tooling

#### Buckets
1. **High-uncertainty contexts**
   - teacher entropy is high
2. **Ambiguous contexts**
   - teacher top-1 and top-2 close
3. **Top-K truncation failures**
   - gold token just outside Top-K

#### For each case, store
- context
- gold token
- teacher top predictions
- baseline method predictions
- proposed method predictions

#### Goal
Show when:
- Top-K becomes overconfident
- sampling is noisy
- heuristic tail summaries help
- compact tail representations better preserve alternatives

---

## Evaluation Plan

### Quantitative
Main metrics:
- **Negative log-likelihood (NLL)**
- **Perplexity (PPL)**
- **Storage per token**
- **Total cache size**

Main comparison:
- performance vs budget

### Qualitative
Inspect representative cases from:
- high-uncertainty contexts
- ambiguous-token contexts
- Top-K truncation failures

We want to understand:
- whether predictions collapse too sharply,
- whether alternative plausible tokens are preserved,
- whether tail-aware methods recover useful low-probability candidates.

### Sub-component analysis
For compact tail representations:
- analyze latent dimensionality vs performance
- optionally analyze reconstruction quality if useful

---

## Main Contribution Framing

We should **not** frame the project as:
- “finding the best sparse KD method overall”
- “solving sparse KD in full generality”

We **should** frame it as:
- comparing better sparse teacher representations under fixed storage constraints,
- especially methods that preserve useful tail information beyond the Top-K portion.

A safe contribution statement is:

- compare dense and sparse teacher representations under a unified per-token budget,
- study whether useful information beyond Top-K can be preserved via simple summaries or learned compact representations,
- evaluate these representations in offline token-level LM distillation.

---