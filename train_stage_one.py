from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim

from args import get_args
from checkpoint_saver import CheckpointEveryNSteps
from dataset import ForgeryNet
from model import RegionSplicerNet


class SplicedRegionPredictor(pl.LightningModule):
    """
    Spliced region predictor.

    args:
    hparams[Namespace] _ parsed cmd parameters
    """

    def __init__(self, hparams):
        super(SplicedRegionPredictor, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = RegionSplicerNet(
            encoder=hparams.encoder,
            pretrained=eval(hparams.pretrained),
            dims=hparams.dims,
            num_class=hparams.num_class,
            freeze_layers=eval(hparams.freeze_layers),
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_dataloader(self):

        dataset = ForgeryNet(
            train_images=self.hparams.dataset_path,
            augmented_images=self.hparams.augmented_dataset_path,
            image_size=self.hparams.input_size,
            manipulation_type=self.hparams.manipulation_type,
            mode="train",
            stage1=True,
            data_augmentation_type=self.hparams.data_augmentation_type,
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=True,
        )

        return loader

    def forward(self, x):
        logits, embeds, features = self.model(x)
        return logits, embeds, features

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            # weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, int(self.hparams.num_epochs)
        )
        return [optimizer], [scheduler]

    def on_train_start(self):
        print("Starting training.")

    def training_step(self, batch, batch_idx):
        x = torch.cat(batch, axis=0)
        y = torch.arange(len(batch))
        y = y.repeat_interleave(len(batch[0])).cuda()
        logits, _, _ = self(x)
        loss = self.criterion(logits, y)
        predicted = torch.argmax(logits, axis=1)
        accuracy = torch.true_divide(torch.sum(predicted == y), predicted.size(0))
        self.log("train_acc", accuracy)
        self.log('train_loss', loss)
        
        return loss

    def on_train_end(self):
        print("Training has ended.")


if __name__ == "__main__":

    args = get_args()

    NAME_CKPT = (
        args.checkpoint_filename + "-" + Path(args.dataset_path).parent.stem
        if args.checkpoint_filename == "weights"
        else args.checkpoint_filename
    )
    logger = TensorBoardLogger(args.log_dir, name=args.log_dir_name)
    checkpoint_dir = (
        Path(logger.save_dir)
        / logger.name
        / f"version_{logger.version}"
        / "checkpoints"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor_checkpoint,
        dirpath=str(checkpoint_dir),
        save_last=False,
        filename=NAME_CKPT,
        mode=args.monitor_checkpoint_mode,
        save_on_train_epoch_end=True,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    model = SplicedRegionPredictor(hparams=args)

    # ___________________________training__________________________________________________________

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        gpus=args.num_gpus,
        callbacks=[
            checkpoint_callback,
            CheckpointEveryNSteps(save_step_frequency=107),
            lr_monitor_callback,
        ],
        max_epochs=int(args.num_epochs),
    )
    trainer.fit(model)
