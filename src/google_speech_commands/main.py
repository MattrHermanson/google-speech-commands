import json
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from google_speech_commands.dataset import AudioClipDataset
from google_speech_commands.model import Classifier

TRAIN_PATH = "./data/train.X"
TRAIN_LABEL_PATH = "./data/train.CT"
DEV_PATH = "./data/dev.X"
DEV_LABEL_PATH = "./data/dev.CT"
CLASS_NAME_PATH = "./data/classes.txt"


@dataclass
class Config:
    input_dim: int = 16000
    num_classes: int = 10
    batch_size: int = 128
    hidden_dim: int = 128
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.5


def log_config(writer: SummaryWriter, cfg: Config) -> None:
    params = asdict(cfg)

    writer.add_text("config", f"```json\n{json.dumps(params, indent=2)}\n```", 0)

    metrics = ["dev_loss", "dev_acc", "dev_macro_f1"]
    exp, ssi, sei = hparams(params, {m: 0.0 for m in metrics})
    file_writer = writer._get_file_writer()
    file_writer.add_summary(exp)
    file_writer.add_summary(ssi)
    file_writer.add_summary(sei)


def main() -> None:
    print("Starting...")
    cfg = Config()

    # load training data
    train = AudioClipDataset(TRAIN_PATH, TRAIN_LABEL_PATH)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)

    # load dev data
    dev = AudioClipDataset(DEV_PATH, DEV_LABEL_PATH)
    dev_loader = DataLoader(dev, batch_size=cfg.batch_size, shuffle=False)

    model = Classifier(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    f1_metric = MulticlassF1Score(num_classes=cfg.num_classes, average="macro")
    acc_metric = MulticlassAccuracy(num_classes=cfg.num_classes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{timestamp}"
    writer = SummaryWriter(run_dir)
    log_config(writer, cfg)

    best_vloss = 1_000_000.0

    for epoch_number in range(cfg.epochs):
        print(f"epoch {epoch_number + 1}:")

        # make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(train_loader, criterion, optimizer, model, epoch_number, writer)

        running_vloss = 0.0
        running_acc = 0.0
        # set the model to evaluation mode, disabling dropout
        model.eval()

        # disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(dev_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss
                running_acc += acc_metric(voutputs, vlabels)
                f1_metric.update(voutputs, vlabels)

        avg_vloss = running_vloss / (i + 1)
        avg_dev_acc = running_acc / (i + 1)
        dev_f1 = f1_metric.compute()

        print(
            f"loss train {avg_loss:.4f} dev {avg_vloss:.4f} | dev acc {avg_dev_acc:.4f} dev f1 {dev_f1:.4f}"
        )

        writer.add_scalar("train_loss", avg_loss, epoch_number + 1)
        writer.add_scalar("dev_loss", avg_vloss, epoch_number + 1)
        writer.add_scalar("dev_acc", avg_dev_acc, epoch_number + 1)
        writer.add_scalar("dev_macro_f1", dev_f1, epoch_number + 1)
        writer.flush()

        f1_metric.reset()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f"{run_dir}/model_best.pt")

    writer.close()


def train_one_epoch(loader, criterion, optimizer, model, epoch_index, tb_writer):
    running_loss = 0.0
    epoch_loss = 0.0

    for i, data in enumerate(loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar("train_loss_batch", last_loss, tb_x)
            running_loss = 0.0

    return epoch_loss / len(loader)


if __name__ == "__main__":
    main()
