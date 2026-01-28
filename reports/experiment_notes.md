# Experiment Notes

## Day 3 — Gradient Statistics (Baseline MLP)

### Setup
- Model: MLP (depth=4, hidden_dim=256)
- Dataset: MNIST
- Optimizer: Adam
- Loss: CrossEntropy
- Metric analyzed: per-parameter gradient norms

### Observations
- Mean gradient norms vary significantly across layers.
- Early layers exhibit smaller average gradients compared to later layers.
- Gradient variance is non-uniform, suggesting uneven learning dynamics.

### Preliminary Insight
Optimization dynamics are layer-dependent even in shallow networks.
This suggests that depth alone does not guarantee uniform gradient flow.

### Interpretation (Day 3)

The classifier layer exhibits significantly higher gradient variance compared to
earlier feature extraction layers. This suggests that learning is concentrated
near the output, while early layers update more conservatively.

Bias parameters consistently show lower variance than weights, indicating that
most learning signal flows through weight matrices rather than additive terms.

These observations confirm that optimization dynamics are highly non-uniform
across depth, even in relatively shallow networks.

### Next Steps
- Track gradient evolution across epochs.
- Study how depth scaling alters these statistics.

