from soundon.Vocoder import BigVGanVocoder
from soundon.AcousticModel import AcousticModel
from dtokenizer.audio.model.hubert_model import HubertTokenizer
from datasets import load_dataset
import torch
import lightning as L

# Initialize the tokenizer and vocoder
ht = HubertTokenizer('hubert_layer6_code100')
vc = BigVGanVocoder()


class HYLeeDataset():
    def __init__(self):
        self.ds = load_dataset("keithito/lj_speech", split='train')

    def __len__(self):
        return len(self.ds)
        # return 1

    def __getitem__(self, idx):
        codec, _ = ht.encode(torch.tensor(self.ds[idx]['audio']['array'], dtype=torch.float),
                             self.ds[idx]['audio']['sampling_rate'])
        code = codec[0]['code']
        mel = vc.get_mel_spectrogram(self.ds[idx]['audio']['array'], self.ds[idx]['audio']['sampling_rate'])
        return torch.tensor(code, dtype=torch.long), torch.tensor(mel, dtype=torch.float)


# Create the dataset
train_dataset = HYLeeDataset()

# Initialize the model
model = AcousticModel(100, train_dataset=train_dataset, batch_size=64, lr=4e-4)

# Initialize the trainer with the custom LearningRateFinder and LearningRateMonitor callbacks
trainer = L.Trainer(
    max_epochs=30,
    accumulate_grad_batches=4,
    enable_progress_bar=True,
    enable_checkpointing=True,
)

# Train the model
trainer.fit(model)

codec, _ = ht.encode(torch.tensor(train_dataset.ds[0]['audio']['array'], dtype=torch.float),
                     train_dataset.ds[0]['audio']['sampling_rate'])
code = codec[0]['code']

# test the trained model
mel = model.generate(torch.tensor([code])).to('cuda')
x = mel[0][0]
border = len(x)
for n, i in enumerate(torch.logical_and(x >= 1, x <= -1)):
    if i == True and border > n:
        border = n
mel = mel[:, :, :border]
wav = vc.generate(mel)

import soundfile as sf
import numpy as np

sf.write('test.wav', np.ravel(wav), 24000)
