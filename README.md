# Foundations of Deep Learning: Optimization, Generalization & Failure Modes

## Motivation
Deep neural networks train and generalize despite extreme overparameterization,
non-convex objectives, and noisy optimization. This project investigates why
training succeeds, when it fails, and how optimization and generalization
interact in modern deep learning systems.

## Key Empirical Findings

### Depth-Dependent Gradient Degradation
![Gradients vs depth](figures/depth_8_feature_extractor.0.weight.png)

Despite successful optimization, deeper networks exhibit early gradient attenuation,
revealing trainability limits that are not reflected in accuracy alone.

---

### Representation Drift under Memorization
![Representation drift](figures/drift_noise_40.png)

With noisy supervision, internal representations drift rapidly even while training
accuracy improves, indicating memorization rather than structured learning.

---

### Gradient Variance and Learning Failure
![Gradient variance](figures/gradvar_noise_40.png)

Increased early-layer gradient variance temporally aligns with accelerated
representation drift, providing a mechanistic explanation for generalization failure.

## Core Research Questions
- How do gradients behave as depth, width, and optimization choices vary?
- Why do some networks train successfully while others fail silently?
- What mechanisms govern generalization under noise and overfitting regimes?

## Scope
This project focuses on controlled empirical studies of:
- Optimization dynamics and trainability
- Gradient behavior in deep networks
- Generalization and memorization
- Failure modes beyond accuracy metrics

The emphasis is on mechanistic understanding rather than benchmark performance.

## Repository Structure

