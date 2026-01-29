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

## Day 9 — Representation Drift as a Signature of Memorization

### Setup
- Model: MLP, depth = 8
- Optimizer: Adam
- Initialization: He
- Comparison: clean labels (0% noise) vs noisy labels (40%)
- Representations extracted from the final hidden layer
- Drift measured via mean cosine similarity to epoch-0 representations
- Analysis performed on a random subset of 2,000 samples for computational stability

### Observation
In the clean-label setting, representations evolve gradually and retain high cosine
similarity to their initial configuration across epochs.

Under 40% label noise, representations drift significantly faster, with a sharp
drop in similarity during early training epochs followed by continued divergence.

Importantly, this drift occurs even though optimization remains stable and training
accuracy continues to increase.

### Interpretation
Memorization is characterized not by optimization failure, but by accelerated
reorganization of the internal feature space. The network continues to update
representations aggressively, but these updates no longer correspond to meaningful
structure in the data.

### Insight
Generalization failure manifests as **representational instability**, not gradient
collapse or divergence. Representation drift provides a mechanistic explanation
for how networks transition from learning to memorization under noisy supervision.

## Day 10 — Gradient Variance as a Driver of Representational Instability

### Observation
Under label noise, early-layer gradient variance increases substantially during
early training epochs. This increase occurs despite stable optimization and
continued improvement in training accuracy.

The temporal rise in gradient variance aligns closely with the onset of rapid
representation drift observed in the hidden feature space.

### Interpretation
Elevated gradient variance does not destabilize optimization directly, but it
induces aggressive and unstructured updates to internal representations.
These updates accelerate representational drift, pushing the network toward
memorization rather than structured learning.

### Insight
Optimization stability, gradient health, and representation stability are
distinct but coupled phenomena. A network may train successfully while its
internal representations become progressively less meaningful.

### Takeaway
Generalization failure under noisy supervision can be understood as a consequence
of gradient-induced representational instability, rather than as an optimization
breakdown.

## Day 11 — Width Scaling and Non-Monotonic Generalization

### Setup
- Architecture: MLP
- Fixed depth: 8
- Hidden widths: 64, 128, 256, 512
- Label noise: 40%
- Optimizer: Adam
- Initialization: He
- Dataset: MNIST
- Training protocol unchanged from previous experiments

### Observation
Increasing model width improves optimization efficiency, as reflected by faster
convergence and higher training accuracy. However, test accuracy does not improve
proportionally under noisy supervision.

The train–test generalization gap exhibits a non-monotonic relationship with width:
it initially increases and then plateaus or worsens for larger widths.

### Interpretation
Overparameterization enables models to fit noisy labels more effectively without
learning robust features. While increased width reduces optimization difficulty,
it does not mitigate memorization-induced generalization failure.

### Insight
Model capacity alone is insufficient to resolve learning under corrupted supervision.
Width scaling amplifies memorization dynamics rather than stabilizing generalization,
highlighting a decoupling between optimization and robust feature learning.

## Day 12 — Dataset Shift Reveals Representation Fragility

### Setup
- Architecture: MLP (depth 8, width 256)
- Dataset: MNIST
- Distribution shift: Random rotation (±30°) applied only at test time
- Label noise levels: 0% and 40%
- Optimizer: Adam
- Epochs: 15

### Observations
Under clean supervision (0% noise), the model achieves near-perfect training
accuracy (~99%) and maintains strong performance under dataset shift
(~90–91%).

Under noisy supervision (40% noise), training accuracy saturates at a much lower
value (~60–61%), and performance under dataset shift drops further
(~84–87%) with noticeable instability across epochs.

Despite the noisy model not fully memorizing the training data, it generalizes
significantly worse under distribution shift.

### Interpretation
Noisy supervision degrades the quality of learned representations rather than
just final accuracy. The model fails to capture invariant features that remain
stable under input transformations.

Dataset shift exposes representational brittleness that is not apparent from
training accuracy alone.

### Key Insight
Generalization robustness depends on representation quality, not optimization
success. Label noise induces shortcut learning that collapses under even mild
distribution shifts.

### Takeaway
Dataset shift serves as a diagnostic tool for distinguishing true feature
learning from superficial fitting, revealing hidden failure modes caused by
noisy supervision.

