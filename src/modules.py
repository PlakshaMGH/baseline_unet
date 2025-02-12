import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.aggregation import MeanMetric


class InstrumentsUNetModel(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.Unet(
            encoder_name, 5, "imagenet", in_channels, classes=out_classes, **kwargs
        )
        self.loss = torch.nn.BCEWithLogitsLoss()

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.dice_loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self.ce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.iou_metric = BinaryJaccardIndex()
        self.val_mean_iou = MeanMetric()

        # freeze the encoder weights
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.test_best_avg_iou = 0.0

    def forward(self, image):
        # normalize the image
        x = (image - self.mean) / self.std
        return self.model(x)

    def shared_step(self, batch, stage):
        image = batch[0]
        mask = batch[1]

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        dice_loss = self.dice_loss_fn(logits_mask, mask)
        ce_loss = self.ce_loss_fn(logits_mask, mask)
        loss = dice_loss + ce_loss

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        iou = self.iou_metric(pred_mask, mask)

        return {
            "total_loss": loss,
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "iou": iou,
        }

    def shared_epoch_end(self, outputs, stage):
        # compute the average of the metrics
        avg_total_loss = torch.stack([x["total_loss"] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x["dice_loss"] for x in outputs]).mean()
        avg_ce_loss = torch.stack([x["ce_loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["iou"] for x in outputs]).mean()

        self.log_dict(
            {
                f"{stage}/avg_total_loss": avg_total_loss,
                f"{stage}/avg_dice_loss": avg_dice_loss,
                f"{stage}/avg_ce_loss": avg_ce_loss,
                f"{stage}/avg_iou": avg_iou,
            }
        )

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)

        self.log_dict(
            {
                f"total_loss": train_loss_info["total_loss"],
                f"iou": train_loss_info["iou"],
            },
            prog_bar=True,
        )

        return train_loss_info["total_loss"]

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def on_train_end(self):
        self.loggers[0].log_metrics({"test/best_avg_iou": self.test_best_avg_iou})
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "test")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "test")
        self.validation_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return {
            "optimizer": optimizer,
        }
