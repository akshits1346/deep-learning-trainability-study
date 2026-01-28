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

## Day 6 — Initialization Schemes and Gradient Stability

### Setup
- Depth fixed at 8 (challenging regime)
- Initialization schemes: Xavier, He, Orthogonal
- Aggregated mean gradient norms per epoch

### Observation
Initialization choice significantly affects early training dynamics.
He initialization maintains higher gradient magnitudes in early layers during
initial epochs, while Xavier exhibits faster attenuation.

Orthogonal initialization shows more stable gradient propagation initially,
but does not fully prevent gradient decay over training.

### Insight
Proper initialization can delay—but not eliminate—trainability degradation
in deeper networks. Initialization interacts strongly with depth but does not
override fundamental optimization limitations.

### Implication
Initialization schemes modulate gradient flow primarily during early training,
suggesting that trainability bottlenecks emerge from deeper structural factors
beyond initialization alone.

## Day 7 — Optimizer Dynamics and Gradient Variance

### Setup
- Depth fixed at 8
- Initialization: He
- Optimizers compared: SGD, SGD with momentum, Adam
- Epoch-aggregated gradient mean and variance tracked

### Observation
SGD exhibits rapid decay of gradient magnitudes in early layers, accompanied by
high variance and unstable training dynamics.

Momentum partially stabilizes gradient flow, reducing variance but not fully
preventing attenuation.

Adam maintains higher and more stable gradient magnitudes across epochs,
particularly in early layers, with narrower variance bands.

### Insight
Adaptive optimizers such as Adam reshape gradient statistics, reducing apparent
gradient collapse. However, this stabilization may mask underlying optimization
pathologies rather than eliminate them.

### Implication
Optimizer choice significantly affects observed trainability, suggesting that
trainability assessments must be interpreted jointly with optimizer dynamics.

## Day 8 — Generalization vs Memorization under Label Noise

### Setup
- Depth fixed at 8
- Initialization: He
- Optimizer: Adam
- Label noise fractions: 0%, 20%, 40%

### Observation
With increasing label noise, training accuracy remains high while test accuracy
degrades substantially, indicating memorization of corrupted labels.

Despite poor generalization, optimization remains stable and gradients do not
collapse immediately, particularly under Adam.

The gap between training and test accuracy widens with noise, even though loss
continues to decrease.

### Insight
Optimization success does not imply generalization. Networks can maintain stable
training dynamics while shifting from learning meaningful patterns to memorizing
noise.

### Implication
Trainability and generalization are partially decoupled phenomena; analyzing
optimization alone is insufficient to assess model reliability.

