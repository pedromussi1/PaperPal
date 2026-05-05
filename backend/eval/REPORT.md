# PaperPal eval report

_Generated 2026-05-05 14:27:08_

- Dataset: `dataset.jsonl` (12 questions on the Transformer paper)
- RAG run: `rag-mvp`
- Baseline run: `baseline-mvp` (no retrieval)

## Aggregate metrics

| Metric | No-RAG baseline | RAG | Lift |
|---|---:|---:|---:|
| Citation precision | 0.000 | 0.500 | +0.500 |
| Citation recall    | 0.000 | 0.625 | +0.625 |
| Citation F1        | 0.000 | 0.542 | +0.542 |
| Mean latency (s)   | 1.48 | 1.71 | +0.23 |

**How to read this:** the citation-accuracy metrics measure whether the model's `[paper_id:page]` citations point at pages where the answer actually lives. The no-RAG baseline cannot produce real citations (it has no retrieval), so its scores are near zero — the gap is the *lift* retrieval gives.

## Per-question breakdown (RAG)

| ID | Category | Cited | Expected | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|
| `att-01` | explanation | [2, 8] | [2] | 0.50 | 1.00 | 0.67 |
| `att-02` | factoid | [3] | [3] | 1.00 | 1.00 | 1.00 |
| `att-03` | formula | [] | [4] | 0.00 | 0.00 | 0.00 |
| `att-04` | factoid | [3] | [5] | 0.00 | 0.00 | 0.00 |
| `att-05` | explanation | [5] | [5] | 1.00 | 1.00 | 1.00 |
| `att-06` | explanation | [6, 8] | [6] | 0.50 | 1.00 | 0.67 |
| `att-07` | factoid | [6] | [6] | 1.00 | 1.00 | 1.00 |
| `att-08` | factoid | [7, 8] | [7] | 0.50 | 1.00 | 0.67 |
| `att-09` | factoid | [8] | [7] | 0.00 | 0.00 | 0.00 |
| `att-10` | factoid | [8] | [8] | 1.00 | 1.00 | 1.00 |
| `att-11` | factoid | [7] | [8] | 0.00 | 0.00 | 0.00 |
| `att-12` | synthesis | [6, 7] | [2, 7] | 0.50 | 0.50 | 0.50 |

## Side-by-side examples

### att-01 — explanation

**Q:** What is the core mechanism behind the Transformer architecture, and what does it replace?

**Expected pages:** [2]

**Gold answer:** Self-attention. The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output, without using sequence-aligned RNNs or convolution.


**RAG answer:**

> The core mechanism behind the Transformer architecture is self-attention [bdfaa68d8984f0dc:2], an attention mechanism relating different positions of a single sequence to compute its representation. Self-attention has been used in conjunction with recurrent networks [bdfaa68d8984f0dc:8]. The Transformer, however, relies entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution [bdfaa68d8984f0dc:2].


**No-RAG baseline answer:**

> The core mechanism behind the Transformer architecture is self-attention, which allows the model to weigh the importance of different input elements relative to each other. This is done through a process called dot-product attention.
> 
> Self-attention replaces traditional recurrent neural networks (RNNs) or convolutional neural networks (CNNs), which rely on sequential processing and fixed-size context windows, respectively. The Transformer architecture uses self-attention to capture long-range dependencies in the input data without relying on sequential processing, making it more parallelizable and efficient for tasks like machine translation and text summarization.

### att-02 — factoid

**Q:** How many layers are in the encoder stack of the base Transformer model?

**Expected pages:** [3]

**Gold answer:** The encoder is a stack of N = 6 identical layers.


**RAG answer:**

> The encoder stack of the base Transformer model is composed of N = 6 identical layers [bdfaa68d8984f0dc:3].


**No-RAG baseline answer:**

> The base Transformer model has 6 layers in its encoder stack.

### att-03 — formula

**Q:** What is the formula for Scaled Dot-Product Attention?

**Expected pages:** [4]

**Gold answer:** Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.


**RAG answer:**

> The formula for Scaled Dot-Product Attention is:
> 
> Attention(Q, K, V) = softmax(QKT / √dk)V
> 
> [1]


**No-RAG baseline answer:**

> The Scaled Dot-Product Attention (SDPA) formula is:
> 
> QK^T / sqrt(d)
> 
> where Q and K are query and key matrices, respectively, d is the dimensionality of the vectors, and ^T denotes matrix transpose.
> 
> This is a simplified version of the original formula. In practice, you may also see additional terms such as a softmax function applied to the result.
