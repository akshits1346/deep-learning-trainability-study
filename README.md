# Foundations of Deep Learning: Optimization, Generalization & Failure Modes

## Motivation
Deep neural networks train and generalize despite extreme overparameterization,
non-convex objectives, and noisy optimization. This project investigates *why*
training succeeds, *when* it fails, and *how* optimization and generalization
interact in modern deep learning systems.

## Core Research Questions
- How do gradients behave as depth, width, and optimization choices vary?
- Why do some networks train successfully while others fail silently?
- What mechanisms govern generalization under noise and overfitting regimes?

## Scope
This project focuses on controlled empirical studies of:
- Optimization dynamics (gradient norms, variance, stability)
- Trainability limits in deep networks
- Generalization and memorization behavior
- Representation dynamics during training

We emphasize *mechanistic understanding* over benchmark chasing.

## Methodology Overview
- Minimal neural architectures (MLPs, small CNNs)
- Controlled synthetic and real datasets
- Systematic variation of depth, initialization, noise, and optimizers
- Reproducible experiments with detailed logging

## Repository Structure

