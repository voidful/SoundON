import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from torch.nn.utils.rnn import pad_sequence


class AcousticModel(L.LightningModule):
    def __init__(self, codebook_size: int = 100, train_dataset=None, batch_size: int = 32):
        super().__init__()
        self.encoder = Encoder(codebook_size)
        self.decoder = Decoder()
        self.train_dataset = train_dataset
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)

    def compute_loss(self, mels_: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        mels_ = mels_.transpose(1, 2)
        padded_mels = torch.full_like(mels_, -100.0)
        padded_mels[:, :, :mels.size(-1)] = mels
        loss_fn = FocalL1Loss()
        loss = loss_fn(mels_, padded_mels)
        self.log('loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, mels, mels_lengths = batch
        mels_ = self.forward(x)
        return self.compute_loss(mels_, mels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 - (epoch / 100)  # Linear decay
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=self.collate_fn)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)

    @staticmethod
    def collate_fn(batch):
        xs, mels, mels_lengths = zip(*batch)
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
        mels_lengths = torch.tensor([mel.size(2) for mel in mels], dtype=torch.long)
        max_len = max(mels_lengths).item()
        max_mel_size = max([mel.size(1) for mel in mels])
        padded_mels = torch.full((len(mels), max_mel_size, max_len), -100.0).to(mels[0].device)
        for i, mel in enumerate(mels):
            padded_mels[i, :, :mel.size(2)] = mel
        return xs, padded_mels, mels_lengths


class Encoder(nn.Module):
    def __init__(self, codebook_size: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size + 1, 256)
        self.prenet = PreNet(256, 256, 256)
        self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 3, 1),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class PreNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size, time_steps, input_size = x.shape
            x = x.view(batch_size * time_steps, input_size)
        elif x.dim() == 2:
            batch_size, input_size = x.shape
            time_steps = 1
        else:
            raise ValueError("Unexpected input dimension: {}".format(x.dim()))
        x = self.net(x)
        x = x.view(batch_size, time_steps, -1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, 100, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        batch_size = xs.size(0)
        device = xs.device

        h1, c1 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)
        h2, c2 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)
        h3, c3 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            x = x.unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1).transpose(1, 2)


class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        l1_loss = F.l1_loss(input, target, reduction='none')
        focal_weight = torch.pow(1 - torch.exp(-l1_loss), self.gamma)
        focal_l1_loss = focal_weight * l1_loss

        if self.reduction == 'mean':
            return focal_l1_loss.mean()
        elif self.reduction == 'sum':
            return focal_l1_loss.sum()
        else:
            return focal_l1_loss
