from soundon.Vocoder import BigVGanVocoder
from soundon.AcousticModel import AcousticModel
from asrp.voice2code_model import hubert
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from datasets import load_dataset

ds = load_dataset("ky552/ML2021_ASR_ST")

print(ds['train'][231])

# prepare training data
# create a vocoder
vocoder = BigVGanVocoder()
mel = vocoder.get_mel_spectrogram(ds['train'][231]['audio']['array'], ds['train'][231]['audio']['sampling_rate'])
# generate hubert code
hubert_model = hubert.hubert_layer6_code100(batch=10, chunk_sec=100)
hubert_dict = hubert_model(input_values=[torch.tensor(ds['train'][231]['audio']['array'], dtype=torch.float32)])

# initialize the acoustic model
am = AcousticModel()


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]['code']
        mel = self.data[idx]['mel']
        mel_length = mel.shape[-1]
        return torch.tensor(code, dtype=torch.long), torch.tensor(mel, dtype=torch.float), torch.tensor(mel_length,
                                                                                                        dtype=torch.long)


train_data = [{'code': hubert_dict['code'], 'mel': mel[0]}]
validation_data = [{'code': hubert_dict['code'], 'mel': mel[0]}]
train_dataset = MyDataset(train_data)
validation_dataset = MyDataset(validation_data)

model = AcousticModel(train_dataset=train_dataset, validation_dataset=validation_dataset, batch_size=1)
trainer = L.Trainer(max_epochs=1000)
trainer.fit(model)

# test the trained model
mel = model.generate(torch.tensor([a['merged_code']]))
x = mel[0][0]
border = len(x)
for n, i in enumerate(torch.logical_and(x >= -101, x <= -99)):
    if i == True and border > n:
        border = n
mel = mel[:, :, :border]
wav = vocoder.generate(mel)

# save the generated wav
import soundfile as sf
import numpy as np

sf.write('example.wav', np.ravel(wav), 24000)
