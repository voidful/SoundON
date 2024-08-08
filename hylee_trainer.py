from soundon.Vocoder import BigVGanVocoder
from soundon.AcousticModel import AcousticModel
from dtokenizer.audio.model.hubert_model import HubertTokenizer
from datasets import load_dataset
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder

# Custom LearningRateFinder for fine-tuning
class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

# Initialize the tokenizer and vocoder
ht = HubertTokenizer('zh_hubert_layer20_code2000')
vc = BigVGanVocoder()

class HYLeeDataset():
    def __init__(self):
        self.ds = load_dataset("WeiChihChen/fixed-ml2021-hungyi-corpus", split='test')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        codec, _ = ht.encode(torch.tensor(self.ds[idx]['audio']['array'], dtype=torch.float),
                             self.ds[idx]['audio']['sampling_rate'])
        code = codec[0]['code']
        mel = vc.get_mel_spectrogram(self.ds[idx]['audio']['array'], self.ds[idx]['audio']['sampling_rate'])
        mel_length = mel.shape[-1]
        return torch.tensor(code, dtype=torch.long), torch.tensor(mel, dtype=torch.float), torch.tensor(mel_length,
                                                                                                        dtype=torch.long)

# Create the dataset
train_dataset = HYLeeDataset()

# Initialize the model
model = AcousticModel(2000, train_dataset=train_dataset, batch_size=32, lr=3e-3)

# Initialize the trainer with the custom LearningRateFinder and LearningRateMonitor callbacks
trainer = L.Trainer(
    max_epochs=20,
    enable_progress_bar=True,
    enable_checkpointing=True,
    callbacks=[
        FineTuneLearningRateFinder(milestones=(5, 10)),
        LearningRateMonitor(logging_interval='step')
    ]
)

# Train the model
trainer.fit(model)
