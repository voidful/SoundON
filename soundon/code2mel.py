# ref: https://github.com/bshall/acoustic-model/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L


class AcousticModel(L.LightningModule):
    def __init__(self, codebook_size: int = 100, upsample: bool = True, train_dataset=None, validation_dataset=None,
                 batch_size: int = 32):
        super().__init__()
        self.encoder = Encoder(codebook_size, upsample)
        self.decoder = Decoder()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x, mels)

    def compute_loss(self, mels_: torch.Tensor, mels: torch.Tensor, mels_lengths: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
        loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        x, mels, mels_lengths = batch
        mels_ = self.forward(x, mels)
        return self.compute_loss(mels_, mels, mels_lengths)

    def validation_step(self, batch, batch_idx):
        x, mels, mels_lengths = batch
        mels_ = self.forward(x, mels)
        return self.compute_loss(mels_, mels, mels_lengths)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class Encoder(nn.Module):
    def __init__(self, codebook_size: int = 100, upsample: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size + 1, 256)
        self.prenet = PreNet(256, 256, 256)
        self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
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


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(128, 256, 256)
        self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, 128, bias=False)

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        mels = self.prenet(mels)
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
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

        m = torch.zeros(batch_size, 128, device=device)
        h1, c1 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)
        h2, c2 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)
        h3, c3 = torch.zeros(1, batch_size, 768, device=device), torch.zeros(1, batch_size, 768, device=device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


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
        return self.net(x)


class DurPerditionNet(nn.Module):
    def __init__(
            self,
            encoder_embed_dim=128,
            var_pred_hidden_dim=128,
            var_pred_kernel_size=3,
            var_pred_dropout=0.5
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2
            ),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size, padding=1
            ),
            nn.ReLU()
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln1(x), p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)
