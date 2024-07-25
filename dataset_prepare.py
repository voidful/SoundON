import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

BIGVGAN_PARAMS = {
    'sampling_rate': 24000,
    'n_fft': 1024,
    'win_size': 1024,
    'hop_size': 256,
    'num_mels': 100,
    'fmin': 0,
    'fmax': None,  # Assuming None means using the Nyquist frequency
}


class MelDataset(Dataset):
    def __init__(self, directory, params):
        self.directory = directory
        self.filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
        self.params = params
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sampling_rate'],
            n_fft=params['n_fft'],
            win_length=params['win_size'],
            hop_length=params['hop_size'],
            n_mels=params['num_mels'],
            f_min=params['fmin'],
            f_max=params['fmax'],
            power=2.0,  # Assuming power=2 for power spectrogram
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.params['sampling_rate']:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.params['sampling_rate'])(
                waveform)
        mel_spec = self.transform(waveform)
        return mel_spec


# Example usage
if __name__ == "__main__":

    dataset = MelDataset(directory='/path/to/audio/files', params=params)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for mel_spec in dataloader:
        print(mel_spec.shape)  # Example processing
