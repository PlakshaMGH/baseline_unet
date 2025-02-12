import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class InstrumentsUNetModel(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.Unet(encoder_name,5,"imagenet", in_channels, classes=out_classes, **kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss()

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.dice_loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) 
        self.ce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # freeze the encoder weights
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.test_best_avg_iou  =  0.0


    
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

        dice_loss =self.dice_loss_fn(logits_mask, mask) 
        ce_loss = self.ce_loss_fn(logits_mask, mask)
        loss = dice_loss + ce_loss

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        metrics = { f"{stage}_mean_image_iou": per_image_iou }

        self.log_dict(metrics, prog_bar=True)

        if stage == "test":
            if per_image_iou > self.test_best_avg_iou:
                self.test_best_avg_iou = per_image_iou

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

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