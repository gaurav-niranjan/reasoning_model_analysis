# Thinking Models Experiments

This folder contains experiments with Qwen3VL Thinking models.

## Overview

Using thinking models, the model reasons and then answers. Next token prediction is not possible like the Instruction tuned models, as the answer comes after the reasoning trace, and the answer token is conditioned on the reasoning trace—which can be different across runs:

```
P(Ans | Q, Reasoning)
```

## Evaluation

We use an LLM as a judge to determine whether the reasoning trace contains mention of the background in any way.