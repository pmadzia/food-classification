"""
Contains functions for training and testing Pytorch model.
"""
import torch

from tqdm import tqdm

def train_step(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: callable,
        device: torch.device
) -> tuple[float, float]:
    """Perform a single training step (one full pass over the training data).

    Args:
        model: The neural network model to be trained.
        data_loader: DataLoader providing the training dataset in batches.
        loss_fn: Loss function used to compute the error between predictions and targets.
        optimizer: Optimizer used to update model parameters.
        accuracy_fn: Function to calculate accuracy. 
            Should accept `y_true` (labels) and `y_pred` (predicted labels).
        device: Target device to run computations on (e.g., 'cpu' or 'cuda').

    Returns:
        Average training loss over all batches and average training accuracy over all batches.
    """
    train_loss, train_acc = 0, 0
    
    model.train()
    for X, y in data_loader:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fn: callable,
        device: torch.device
) -> tuple[float, float]:
    """Perform a single evaluation step (one full pass over the test/validation data).

    Args:
        model: The neural network model to be evaluated.
        data_loader: DataLoader providing the test/validation dataset in batches.
        loss_fn: Loss function used to compute the error between predictions and targets.
        accuracy_fn: Function to calculate accuracy. 
            Should accept `y_true` (labels) and `y_pred` (predicted labels).
        device: Target device to run computations on (e.g., 'cpu' or 'cuda').

    Returns:
        Average test loss over all batches and average test accuracy over all batches.
    """
    test_loss, test_acc = 0, 0
    
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y)
            test_loss += loss
            test_acc += accuracy_fn(y_true=y,
                                     y_pred=test_pred.argmax(dim=1))

        # Adjust metrics to get average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
        return test_loss, test_acc
    
def train(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        accuracy_fn: callable,
        epochs: int,
        device: torch.device
) -> dict[str, list[float]]:
    """Train and evaluate a PyTorch model over multiple epochs.

    Runs the training loop using `train_step` and evaluation loop using `test_step`
    for a specified number of epochs. Tracks and returns loss and accuracy metrics
    for both training and testing.

    Args:
        model: The neural network model to be trained and evaluated.
        train_dataloader: DataLoader providing the training dataset in batches.
        test_dataloader: DataLoader providing the test/validation dataset in batches.
        optimizer: Optimizer used to update model parameters based on gradients.
        loss_fn: Loss function used to compute the error between predictions and targets.
        accuracy_fn: Function to calculate accuracy. 
            Should accept `y_true` (labels) and `y_pred` (predicted labels).
        epochs: Number of epochs (full passes through the training dataset).
        device: Target device to run computations on (e.g., 'cpu' or 'cuda').

    Returns:
        A dictionary containing per-epoch metrics.
    """
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)
        
        print(
            f'Epoch: {epoch+1} | '
            f'train_loss: {train_loss:.4f} | '
            f'train_acc: {train_acc:.4f} | '
            f'test_loss: {test_loss:.4f} | '
            f'test_acc: {test_acc:.4f}'
        )

        results['train_loss'].append(train_loss.item() if hasattr(train_loss, 'item') else train_loss)
        results['train_acc'].append(train_acc.item() if hasattr(train_acc, 'item') else train_acc)
        results['test_loss'].append(test_loss.item() if hasattr(test_loss, 'item') else test_loss)
        results['test_acc'].append(test_acc.item() if hasattr(test_acc, 'item') else test_acc)

    return results