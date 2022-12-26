from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from MiSeCom import MiSeComDataLoader, LitMiSeCom

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_missing_sentence_component")

    hyperparameter = {
        'pretrained_ck': 'roberta-base',
        'method_for_layers': 'mean',
        'layers_use_from_last': 2,
        'lr': 2e-5
    }
    lit_misecom = LitMiSeCom(**hyperparameter)

    # dataloader
    misecom_dataloader = MiSeComDataLoader(pretrained_ck='roberta-base', max_length=25)
    [train_dataloader, test_dataloader, valid_dataloader] = misecom_dataloader.get_dataloader(batch_size=128, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=20, devices=[1], accelerator="gpu", logger=wandb_logger, log_every_n_steps=20)#, strategy="ddp")
    trainer.fit(model=lit_misecom, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_misecom.export_model('misecom_model/v1')
