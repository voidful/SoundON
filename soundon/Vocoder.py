import librosa
import torch

from soundon.vocoder.bigvgan import BigVGAN
from soundon.vocoder.meldataset import get_mel_spectrogram


class BigVGanVocoder:
    BIGVGAN_PARAMS = {
        'sampling_rate': 24000,
        'n_fft': 1024,
        'win_size': 1024,
        'hop_size': 256,
        'num_mels': 100,
        'fmin': 0,
        'fmax': None,  # Assuming None means using the Nyquist frequency
    }

    def __init__(self):
        gan_model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
        gan_model.remove_weight_norm()
        self.gan_model = gan_model.eval().to('cuda')

    def get_mel_spectrogram(self, wav, sr=None):
        if isinstance(wav, str):
            wav, sr = librosa.load(wav, sr=self.gan_model.h.sampling_rate, mono=True)
        else:
            if sr is None:
                # assert sr is required
                raise ValueError("sr must be provided if wav is not a path")
            # convert sr
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.gan_model.h.sampling_rate)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        return get_mel_spectrogram(wav, self.gan_model.h).to('cuda')

    def generate(self, mel):
        with torch.inference_mode():
            wav_gen = self.gan_model(mel)  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
        wav_gen_float = wav_gen.squeeze(0).cpu()  # wav_gen is FloatTensor with shape [1, T_time]
        # you can convert the generated waveform to 16 bit linear PCM
        wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype(
            'int16')  # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype
        return wav_gen_int16
