
# SoundOn: Any Codec to Mel Spectrogram

SoundOn is a project that converts audio files to mel spectrograms, a format suitable for training sound generation models. This repository contains scripts to prepare the dataset and convert audio files to mel spectrograms.

## Getting Started

### Dependencies

- Python 3.8 or later
- PyTorch
- torchaudio
- librosa (optional, for additional audio processing)

Ensure you have Python installed on your system. This project is developed and tested on macOS, but it should be compatible with Linux and Windows, provided all dependencies are met.

### Installing

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/soundon.git
cd soundon
```

### Preparing the Dataset

Place your `.wav` audio files in a directory. Update the `directory` parameter in the `MelDataset` instantiation in `dataset_prepare.py` to point to this directory.

### Running the Dataset Preparation

Execute `dataset_prepare.py` to start the dataset preparation process:

```bash
python dataset_prepare.py
```

This script will process all `.wav` files in the specified directory, converting them into mel spectrograms and printing their shapes.

## Usage

The `MelDataset` class can be used as follows:

```python
from dataset_prepare import MelDataset
from torch.utils.data import DataLoader

params = {
    'sampling_rate': 24000,
    'n_fft': 1024,
    'win_size': 1024,
    'hop_size': 256,
    'num_mels': 100,
    'fmin': 0,
    'fmax': None,
}

dataset = MelDataset(directory='/path/to/audio/files', params=params)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for mel_spec in dataloader:
    print(mel_spec.shape)
```

## Contributing

Contributions to the project are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- The `code2mel.py` and `dataset_prepare.py` scripts are foundational to this project, enabling the conversion of audio files to a format suitable for sound generation models.
```

This `README.md` template provides a comprehensive overview of your project, including how to get started, use the project, and contribute. Adjust the repository URL and any specific details as necessary.