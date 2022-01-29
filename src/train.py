import torch
import torch.optim as optim
import multiprocessing


from dataset import HeadposeDataset, DatasetFromSubset
from datetime import datetime
from model import FSANet

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import TensorBoardLogger
from transforms import Normalize, SequenceRandomTransform, ToTensor

from torch import onnx
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from torchmetrics import Metric, Accuracy

class HeadPoseModel(LightningModule):
    def __init__(self, use_variance = False):
        super().__init__()
        self.fsanet = FSANet(use_variance)
        self.learning_rate = 0.001
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, x):
        return self.fsanet(x)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def convert_to_onnx(model, filename):
    model.eval()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = model(x)
    onnx.export(model, x, filename, export_params=True, do_constant_folding=True,
                input_names = ['input'], output_names=['output'],
                dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} )
    return

def main():
    data_path = '../data/type1/train'
    dataset = HeadposeDataset(data_path, transform=None)

    train_transform = transforms.Compose([
            SequenceRandomTransform(),
            Normalize(mean=127.5, std=128.0),
            ToTensor()
            ])

    validation_transform = transforms.Compose([Normalize(mean=127.5,std=128),ToTensor()])

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_subset, validation_subset = random_split(dataset, [train_size, validation_size])

    train_dataset = DatasetFromSubset(train_subset, train_transform)
    validation_dataset = DatasetFromSubset(validation_subset, validation_transform)

    del dataset,train_subset,validation_subset

    print('Train Dataset Length: ',len(train_dataset))
    print('Validation Dataset Length: ',len(validation_dataset))

    batch_size = 16
    num_cpus = multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    logger = TensorBoardLogger('logs', name='HeadPoseModel', log_graph=False)

    num_epochs = 100000
    for use_variance in [False, True]:
        model = HeadPoseModel(use_variance)
        callbacks = [TQDMProgressBar()]
        trainer = Trainer(accelerator="gpu", devices=-1, 
                          max_epochs=num_epochs, logger=logger, callbacks=callbacks,
                          plugins=DDPPlugin(find_unused_parameters=False))
        trainer.fit(model, train_loader, validation_loader)

        scoring_funtion_type = 'var' if use_variance else '1x1'
        filename = f'headpose-{scoring_funtion_type}-iter-{num_epochs}'
        torch.save(model.state_dict(), f'{filename}.pt')
        convert_to_onnx(model, f'{filename}.onnx')


if __name__ == '__main__':
    main()
