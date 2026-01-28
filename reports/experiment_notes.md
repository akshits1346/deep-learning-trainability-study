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

## Day 4 — Epoch-wise Gradient Dynamics

### Setup
- Model: MLP (depth=4, hidden_dim=256)
- Dataset: MNIST
- Optimizer: Adam
- Gradients logged per mini-batch across epochs

### Observation
Epoch-wise gradient plots show significant vertical dispersion within each epoch.
This reflects high intra-epoch gradient variance due to mini-batch stochasticity.

Early layers (e.g., first feature extraction layer) exhibit lower gradient
magnitudes overall, while later layers retain higher-magnitude updates.

Notably, gradient magnitudes fluctuate strongly even when training accuracy
continues to improve smoothly.

### Interpretation
Raw epoch-wise plots capture the *distribution* of gradients rather than a
single epoch-level statistic. This highlights that optimization noise persists
throughout training and is not visible in loss or accuracy curves.

These plots serve as diagnostic evidence rather than final presentation figures.

### Next Step
Aggregate gradients per epoch (mean, variance) to extract cleaner temporal
signals and detect early optimization collapse.

## Day 5 — Epoch-wise Aggregation and Depth Scaling

### Setup
- Depths evaluated: 2, 4, 8
- Aggregated per-epoch gradient statistics (mean and variance)
- Same optimizer, data, and initialization across runs

### Observation
Aggregated curves reveal clear depth-dependent behavior.
Shallower networks (depth=2) maintain stable gradient magnitudes across epochs,
while deeper networks (depth=8) exhibit faster decay and higher variance early
in training.

Early layers in deeper networks show suppressed gradient magnitudes relative to
later layers, with widening variance bands as depth increases.

### Insight
Depth amplifies gradient imbalance and instability, even when training accuracy
appears normal. Aggregated statistics expose trainability limits that are not
visible in raw accuracy metrics.

### Implication
Trainability degradation with depth manifests as early gradient attenuation and
increased variance, suggesting depth-dependent optimization bottlenecks.

