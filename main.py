from pathlib import Path

import click
import torch
from torch._C import device
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=2,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--num_classes",
              default=2,
              type=int,
              help="Specify the number of output classes to be segmented.")
def main(data_directory, exp_directory, epochs, batch_size,num_classes):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3(num_classes)
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CrossEntropyweights = torch.tensor([0.0026,0.0044,0.993]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=CrossEntropyweights)
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)

    # Specify the evaluation metrics

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    num_epochs=epochs,
                    exp_dir=exp_directory,
                    num_classes=num_classes)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()
