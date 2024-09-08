import torch
import lightning as L
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from soundon.Vocoder import BigVGanVocoder
from soundon.AcousticModel import AcousticModel
from dtokenizer.audio.model.hubert_model import HubertTokenizer
from datasets import load_dataset
from lightning.pytorch.callbacks import Callback


# Helper function to save audio
def save_audio(wav, epoch_num, sample_rate=24000, filename_prefix="test_epoch"):
    sf.write(f'{filename_prefix}_{epoch_num}.wav', np.ravel(wav), sample_rate)
    print(f"Audio for epoch {epoch_num} saved.")


# Helper function to save mel spectrogram
def save_mel_spectrogram(mel, epoch_num, filename_prefix="mel_spectrogram_epoch"):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.squeeze().cpu().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram for Epoch {epoch_num}')
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_{epoch_num}.png')
    plt.close()
    print(f"Mel spectrogram for epoch {epoch_num} saved.")


# Dataset class
class HYLeeDataset():
    def __init__(self):
        self.ds = load_dataset("voidful/gen_ai_2024", split='train')

    def __len__(self):
        return 10
        return len(self.ds)

    def __getitem__(self, idx):
        audio_data = torch.tensor(self.ds[idx]['audio']['array'], dtype=torch.float)
        sampling_rate = self.ds[idx]['audio']['sampling_rate']

        code = self.encode_audio(audio_data, sampling_rate)
        mel = self.get_mel_spectrogram(audio_data, sampling_rate)

        return torch.tensor(code, dtype=torch.long), mel

    def encode_audio(self, audio_data, sampling_rate):
        codec, _ = ht.encode(audio_data, sampling_rate)
        return codec[0]['code']

    def get_mel_spectrogram(self, audio_data, sampling_rate):
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        return vc.get_mel_spectrogram(audio_data, sampling_rate)


# Callback for saving audio and mel spectrogram, including Ground Truth Mel spectrogram
class AudioSaverCallback(Callback):
    def __init__(self, tokenizer, vocoder, dataset):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocoder = vocoder
        self.dataset = dataset

    def get_mel_spectrogram(self, audio_data, sampling_rate):
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        return vc.get_mel_spectrogram(audio_data, sampling_rate)

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} ended.")
        audio_data, sampling_rate = self._get_first_sample()
        code = self.tokenizer.encode(audio_data, sampling_rate)[0][0]['code']
        mel = self.get_mel_spectrogram(audio_data, sampling_rate)
        mel = self._generate_mel(pl_module, code, mel)
        gt_mel = self._get_gt_mel(0)  # Get the Ground Truth Mel spectrogram for comparison
        print(f"Generated mel spectrogram for epoch {trainer.current_epoch}.")
        self._process_and_save(trainer.current_epoch, mel, gt_mel)

    def _get_first_sample(self):
        return torch.tensor(self.dataset.ds[0]['audio']['array'], dtype=torch.float), self.dataset.ds[0]['audio'][
            'sampling_rate']

    def _generate_mel(self, model, code, reference):
        # Generate mel spectrogram using the code and the correct reference speech
        mel = model.generate(torch.tensor([code]).to(model.device))
        return self._trim_mel(mel)

    def _trim_mel(self, mel):
        x = mel[0][0]
        border = len(x)
        for n, i in enumerate(torch.logical_and(x >= -99, x <= -100)):
            if i and border > n:
                border = n
        return mel[:, :, :border]

    def _get_gt_mel(self, idx):
        # Get the Ground Truth Mel spectrogram from the dataset
        _, gt_mel = self.dataset[idx]
        return gt_mel

    def _process_and_save(self, epoch_num, mel, gt_mel):
        # Save the generated audio and Mel spectrogram
        wav = self.vocoder.generate(mel)
        save_audio(wav, epoch_num)
        save_mel_spectrogram(mel, epoch_num)

        # Save the Ground Truth Mel spectrogram
        save_mel_spectrogram(gt_mel, epoch_num, filename_prefix="gt_mel_spectrogram_epoch")


# Initialize the tokenizer and vocoder
ht = HubertTokenizer('zh_hubert_layer20_code2000')
vc = BigVGanVocoder()

# Create the dataset
train_dataset = HYLeeDataset()

# Initialize the model
model = AcousticModel(2000, train_dataset=train_dataset, batch_size=16, lr=4e-4)

# Initialize the custom audio saver callback
audio_saver_callback = AudioSaverCallback(ht, vc, train_dataset)

# Initialize the trainer with the custom audio saver callback
trainer = L.Trainer(
    max_epochs=1000,
    accumulate_grad_batches=32,
    enable_progress_bar=True,
    enable_checkpointing=True,
    callbacks=[audio_saver_callback]
)

# Train the model
trainer.fit(model)
