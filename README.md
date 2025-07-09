
# TML25_A2_15
# TML ASSIGNMENT 2  
## Model Stealing in Supervised Learning

**Team:** 15  
**Members:** Hina Lilaram, Javed Akhtar

---

### Overview

This repository contains our solution for Assignment 3 of the Trustworthy Machine Learning course, focused on **Adversarial Robustness in Image Classification**. The objective is to train a ResNet model on a custom dataset using adversarial training techniques (PGD and FGSM) to improve robustness against adversarial attacks, and to evaluate the model’s performance on both clean and adversarial examples.

---


### Files and Descriptions

- `vanilla.ipynb`: Main notebook with all code for data loading, training, evaluation, and submission.
- `final_high_acc_model.pt`: Trained model weights for submission (not included)
- `adversarial_training_progress.png`: Training and validation progress plots.

---

## Features

- **Dataset Handling:** Custom `TaskDataset` class for loading and transforming image data.
- **Data Augmentation:** Uses torchvision transforms for improved generalization.
- **Model Architectures:**  ResNet34
- **Adversarial Training:** Implements Projected Gradient Descent (PGD) and FGSM attacks for robust training.
- **Training Loop:** Balanced adversarial and clean training with dynamic PGD steps and early stopping.
- **Validation:** Evaluates model on clean, PGD, and FGSM adversarial examples.
- **Checkpointing:** Saves best models and training progress.

## Configuration

Model and training parameters can be adjusted in the `Config` class, including:
- Model type (`resnet34` by default)
- Learning rate, batch size, epochs
- Adversarial attack parameters (epsilon, alpha, PGD steps)

---

*For any questions or clarifications, please contact either team member.*