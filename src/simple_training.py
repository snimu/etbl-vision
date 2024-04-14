
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import polars as pl

import data 
import mlp


def train_mlp(
        with_noise: bool = False,
        noise_level: float = 0.05,
        num_layers: int = 5,
        imsize: int = 28*28,
        num_classes: int = 10,
        batch_size: int = 64,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        num_feedback_signals: int = 10,
        device: str = "cpu",
):
    # Load the data
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=data.get_transforms("additivey", noise_level) if with_noise else ToTensor(),
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    model = mlp.MLP(num_layers, imsize, num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    feedback_interval = len(train_dataloader) // num_feedback_signals

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            if (i != 0) and (i % feedback_interval == 0):
                print(f"Epoch {epoch + 1}/{epochs}, Step {i}/{len(train_dataloader)}, Loss: {loss.item()}")

    return model


@torch.no_grad()
def test_model_for_n_tries(model: mlp.MLP, n: int = 1, noise_level: float = 0.05) -> tuple[float, float, float]:
    """
    Test the model. 
    Noise and analyze each image n times, then average the results.
    Returns the average loss, top-1 accuracy, and top-5 accuracy.
    """
    model.eval()
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    total_loss = 0.0
    top_1 = 0
    top_5 = 0

    for i, (images, labels) in enumerate(test_dataloader):
        predictions = []
        for _ in range(n):
            noisy_image = data.add_noise_additivey(images, noise_level)
            outputs = model(noisy_image)
            predictions.append(outputs)
        predictions = torch.stack(predictions).mean(dim=0)
        loss = nn.CrossEntropyLoss()(predictions, labels)
        total_loss += loss.item()
        top_1 += (predictions.argmax(1) == labels).sum().item()
        top_5 += (labels.view(-1, 1) == predictions.topk(5, 1)[1]).sum().item()

    total_loss /= len(test_dataloader) * batch_size
    top_1 /= len(test_dataloader) * batch_size
    top_5 /= len(test_dataloader) * batch_size

    return total_loss, top_1, top_5


@torch.no_grad()
def test_model_no_noise(model: mlp.MLP) -> tuple[float, float, float]:
    """
    Test the model without adding noise to the images.
    Returns the average loss, top-1 accuracy, and top-5 accuracy.
    """
    model.eval()
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    total_loss = 0.0
    top_1 = 0
    top_5 = 0

    for i, (images, labels) in enumerate(test_dataloader):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        top_1 += (outputs.argmax(1) == labels).sum().item()
        top_5 += (labels.view(-1, 1) == outputs.topk(5, 1)[1]).sum().item()

    total_loss /= len(test_dataloader) * batch_size
    top_1 /= len(test_dataloader) * batch_size
    top_5 /= len(test_dataloader) * batch_size

    return total_loss, top_1, top_5


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--savefile", type=str, default="results.csv")

    parser.add_argument("--with_noise", type=int, nargs="+", default=0, help="Whether to add noise to the images")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level for the images")
    parser.add_argument("--num_tries", type=int, default=1, help="Number of tries to test the model")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers in the MLP")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num_feedback_signals", type=int, default=10, help="Number of feedback signals to print")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")

    args = parser.parse_args()

    args.with_noise = [args.with_noise] if isinstance(args.with_noise, int) else args.with_noise
    args.with_noice = [bool(i) for i in args.with_noise]

    return args


def main():
    args = get_args()
    noise_trials = (1, 2, 5, 10)
    for with_noise in args.with_noise:
        print(f"\n\nTraining with noise: {with_noise}\n\n")
        for trial in range(args.num_tries):
            print(f"\n\nTrial {trial + 1}/{args.num_tries}\n\n")
            results = {"with_noise": with_noise, "trial": trial}
            model = train_mlp(
                with_noise=with_noise,
                noise_level=args.noise_level,
                num_layers=args.num_layers,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                num_feedback_signals=args.num_feedback_signals,
                device=args.device,
            )

            print("Testing...")
            if with_noise:
                for n in noise_trials:
                    loss, top_1, top_5 = test_model_for_n_tries(model, n, args.noise_level)
                    print(f"{n=}, {loss=}, {top_1=}, {top_5=}")
                    results[f"loss_{n}"] = loss
                    results[f"top_1_{n}"] = top_1
                    results[f"top_5_{n}"] = top_5
            else:
                loss, top_1, top_5 = test_model_no_noise(model)
                print(f"{loss=}, {top_1=}, {top_5=}")
                for n in noise_trials:
                    results[f"loss_{n}"] = loss
                    results[f"top_1_{n}"] = top_1
                    results[f"top_5_{n}"] = top_5

            df = pl.DataFrame(results)
            if not os.path.exists(args.savefile):
                df.write_csv(args.savefile)
            else:
                with open(args.savefile, "ab") as f:
                    df.write_csv(f, include_header=False)


if __name__ == "__main__":
    main()
