# RLHF with Direct Preference Optimization (DPO)

This repository implements and demonstrates **Direct Preference Optimization (DPO)** : a simplified, reward-free approach to **Reinforcement Learning from Human Feedback (RLHF)**.  
The goal of this mini-project is to understand how human preference data can be used to align a language model’s behavior without training a separate reward model.

---

## Overview

In traditional RLHF, models are trained in three stages:
1. **Supervised Fine-Tuning (SFT)** – learn from human demonstrations  
2. **Reward Modeling (RM)** – train a model to predict human preferences  
3. **Reinforcement Learning (PPO)** – fine-tune the base model to maximize the reward model’s score  

**DPO** simplifies this process by skipping the reward model.  
Instead, it directly adjusts the base model’s log-likelihood to favor *preferred* responses over *rejected* ones.

This makes DPO more efficient and accessible for smaller setups like this one.

---

## Implementation Summary

- **Notebook:** `dpo_training.ipynb`  
- **Base Model:** `Qwen2.5-0.5B-Instruct` (lightweight and Colab-friendly)  
- **Trainer:** HuggingFace `trl.DPOTrainer`  
- **Dataset:** Small synthetic dataset of human preferences (chosen vs rejected)  
- **Goal:** Show how model alignment can emerge from preference data using DPO

---
This project was inspired by Shawhin Talebi’s YouTube tutorial.
