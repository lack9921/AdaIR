import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim, AverageMeter
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    # ===== [Elvis] Validation step =====
    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        # Compute PSNR/SSIM for monitoring
        psnr, ssim, n = compute_psnr_ssim(restored, clean_patch)
        self.log("val_psnr", psnr, on_epoch=True, sync_dist=True)
        self.log("val_ssim", ssim, on_epoch=True, sync_dist=True)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = AdaIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    # ===== [Elvis] Auto-set de_type and create validation dataloader =====
    val_loader = None
    if opt.elvis_mode:
        # Auto-override de_type to Elvis 5 types so _init_elvis_ids() fires
        if opt.de_type == ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance']:
            opt.de_type = ['blur', 'haze', 'lowlight', 'rain', 'snow']
        print("[Elvis] de_type = {}".format(opt.de_type))
        print("[Elvis] Creating validation set (last {} images per type)...".format(opt.elvis_val_last_n))
        valset = AdaIRTrainDataset(opt, elvis_val=True)
        val_loader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                                drop_last=False, num_workers=opt.num_workers)
    
    model = AdaIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
