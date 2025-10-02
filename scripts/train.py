"""
Trains Pytorch image-classification model.
"""

import torch

from data_setup import create_dataloaders
from model import TinyVGG
from engine import train
from utils import accuracy_fn, save_model

from torchvision import transforms

BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
EPOCHS = 5

def main():
    train_dir = 'data/processed/baklava_churros_cheesecake/train'
    test_dir = 'data/processed/baklava_churros_cheesecake/test'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model = TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    train(model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        epochs=EPOCHS,
        device=device)

    save_model(model=model,
            model_name='TinyVGG_model.pth')

if __name__ == "__main__":
    main()