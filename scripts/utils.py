"""
Contains utility functions for Pytorch model training, evaluating, ploting and saving.
"""
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def accuracy_fn(
        y_true: torch.Tensor,
        y_pred: torch.Tensor
) -> float:
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true: Truth labels for predictions.
        y_pred: Predictions to be compared to truth predictions.

    Returns:
        Accuracy value between y_true and y_pred.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def save_model(
        model: torch.nn.Module,
        model_name: str,
        target_dir: str = 'models'
) -> None:
    """Saves a Pytorch model to a target directory.

    Args:
        model: A pytorch model to save.
        target_dir: A directory in which model will be save.
        model_name: A filename for the saved model.
            Should include '.pth' or '.pt' as the file extension.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt'),\
    'model_name should end with ".pth" or ".pt"'
    model_save_path = target_dir_path / model_name

    print(f'Saving model to: {model_save_path}.')
    torch.save(obj=model.state_dict(),
               f=model_save_path)

def plot_loss_and_accuracy_curves(results: dict[str, list[float]]) -> None:
    """Plots loss and accuracy curves based on results of model training.

    Args:
        results: A dictionary containing the results of training with the following keys.
            - 'train_loss': List of training losses per epoch.
            - 'test_loss': List of test losses per epoch.
            - 'train_acc': List of training accuracies per epoch.
            - 'test_acc': List of test accuracies per epoch.
    """
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()