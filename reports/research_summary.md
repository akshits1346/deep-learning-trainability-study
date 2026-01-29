# Memorization, Optimization, and Representation Failure in Deep Neural Networks

## Abstract
Despite strong empirical performance, deep neural networks often generalize poorly
under noisy supervision and distribution shift. This work presents an empirical
study connecting optimization dynamics, gradient behavior, and representation
stability to memorization-driven failure modes. Through controlled experiments on
MNIST, we show that optimization success can coexist with representational
instability, leading to brittle generalization. Our findings highlight the need
to evaluate internal learning dynamics rather than surface-level accuracy alone.

## Motivation
Deep neural networks routinely achieve near-perfect training performance even
under extreme overparameterization and corrupted supervision. However, when and
why such optimization success translates into robust generalization remains
poorly understood.

This project investigates how optimization dynamics, gradient statistics, and
internal representations interact to produce memorization and generalization
failure.

## Experimental Framework
We study multilayer perceptrons trained on MNIST under controlled variations of:
- Label noise (0%–40%)
- Network depth and width
- Optimization dynamics
- Dataset shift at test time

Across all experiments, we log training accuracy, test accuracy, gradient
statistics, and hidden-layer representations to probe learning behavior beyond
surface-level metrics.

## Key Findings

### 1. Optimization Success ≠ Generalization
Networks reliably optimize training loss even under severe label noise. However,
high training accuracy does not imply meaningful generalization or robustness.

### 2. Gradient Dynamics Reveal Instability
Early-layer gradient variance increases significantly under noisy supervision,
indicating unstable and inconsistent feature learning during training.

### 3. Representation Drift Under Noise
Internal representations exhibit rapid drift when trained with noisy labels,
suggesting that learned features fail to stabilize and converge.

### 4. Capacity Amplifies Memorization
Increasing model width improves optimization efficiency but does not reduce
generalization error. Overparameterization accelerates memorization without
improving robustness.

### 5. Dataset Shift Exposes Hidden Failure Modes
Models trained under noisy supervision perform significantly worse under
distribution shift, even when training accuracy remains reasonable. Dataset shift
reveals representation brittleness invisible to standard evaluation.

## Evidence Map

Each empirical claim is supported by targeted experiments and logged metrics:

- **Representation Drift Under Noise**  
  Supported by cosine similarity decay of hidden-layer representations under 40% label noise  
  (see: figures/day9_representation_drift/representation_drift_noise40.png)

- **Gradient Variance and Learning Instability**  
  Early-layer gradient variance increases temporally alongside representation drift  
  (see: figures/day10_gradient_variance/early_gradient_variance_noise40.png)

- **Depth-Dependent Trainability Limits**  
  Gradient attenuation in early layers grows with network depth despite stable optimization  
  (see: figures/day10_gradient_variance/depth8_first_layer_gradient_norm.png)

- **Capacity-Induced Memorization**  
  Increasing width accelerates optimization without reducing generalization error  
  (see: width-scaling experiments, Day 11)

- **Dataset Shift Vulnerability**  
  Models trained under noisy supervision degrade sharply under distribution shift  
  (see: figures/day12_dataset_shift/dataset_shift_noise40.png)

## Conclusion
Generalization failure in deep neural networks arises from representational
collapse rather than optimization failure. Robust learning requires stable,
invariant representations — not merely sufficient capacity or optimization
success.

## Methods (Brief)
All experiments use multilayer perceptrons with He initialization and Adam
optimization. We systematically vary label noise, depth, width, and test-time
distribution shift while logging accuracy, gradient statistics, and hidden-layer
representations. Representation drift is measured via cosine similarity to initial
features, and robustness is evaluated under controlled input transformations.

