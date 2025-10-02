"""
Contains utility functions for Pytorch model training, evaluating and saving.
"""
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
        model: torch.nn.Model,
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