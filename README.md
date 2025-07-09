# Trustworthy Machine Learning - Adversarial Robustness with ResNet

This project implements adversarially robust image classification using PyTorch and ResNet architectures. The code is designed for the Trustworthy Machine Learning course (Assignment 3) and focuses on training, evaluating, and submitting robust models against adversarial attacks.

## Features

- **Dataset Handling:** Custom `TaskDataset` class for loading and transforming image data.
- **Data Augmentation:** Uses torchvision transforms for improved generalization.
- **Model Architectures:** Supports ResNet18, ResNet34, and ResNet50.
- **Adversarial Training:** Implements Projected Gradient Descent (PGD) and FGSM attacks for robust training.
- **Training Loop:** Balanced adversarial and clean training with dynamic PGD steps and early stopping.
- **Validation:** Evaluates model on clean, PGD, and FGSM adversarial examples.
- **Checkpointing:** Saves best models and training progress.
- **Submission:** Prepares and submits model for external robustness evaluation.

## Usage

1. **Install Requirements:**
   - Python 3.7+
   - PyTorch
   - torchvision
   - numpy, matplotlib, tqdm, PIL

2. **Prepare Dataset:**
   - Place your dataset file (e.g., `Train.pt`) in the appropriate directory.

3. **Train the Model:**
   - Run the notebook cells sequentially to train the model with adversarial robustness.

4. **Evaluate and Save:**
   - The notebook saves the best model and checkpoints automatically.

5. **Submission:**
   - Use the provided code to submit your model to the evaluation server.

## Key Files

- `vanilla.ipynb`: Main notebook with all code for data loading, training, evaluation, and submission.
- `final_high_acc_model.pt`: Trained model weights for submission.
- `adversarial_training_progress.png`: Training and validation progress plots.

## Configuration

Model and training parameters can be adjusted in the `Config` class, including:
- Model type (`resnet34` by default)
- Learning rate, batch size, epochs
- Adversarial attack parameters (epsilon, alpha, PGD steps)

## Notes

- Ensure the model output shape is `(1, 10)` for compatibility with the evaluation server.
- Only allowed models: `resnet18`, `resnet34`, `resnet50`.
- For submission, replace the token in the request header with your provided token.

## License

This project is for educational purposes as part of the Trustworthy Machine Learning course.
