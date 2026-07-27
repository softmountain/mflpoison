import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Conv1dEncoder(nn.Module):
    def __init__(self, input_dim, n_filters, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = inputs.float().permute(0, 2, 1)
        for convolution in (self.conv1, self.conv2, self.conv3):
            outputs = convolution(outputs)
            outputs = self.relu(outputs)
            outputs = self.pooling(outputs)
            outputs = self.dropout(outputs)
        return outputs.permute(0, 2, 1)


def masked_softmax(values, valid_lengths):
    if valid_lengths is None:
        return nn.functional.softmax(values, dim=-1)
    positions = torch.arange(values.shape[-1], device=values.device)
    mask = positions.unsqueeze(0) >= valid_lengths.to(values.device).unsqueeze(1)
    return nn.functional.softmax(values.masked_fill(mask, -1e6), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, d_hid=64, d_att=256):
        super().__init__()
        self.query_proj = nn.Linear(d_hid, d_att, bias=False)
        self.key_proj = nn.Linear(d_hid, d_att, bias=False)
        self.bias = nn.Parameter(torch.rand(d_att).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_att, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, valid_lens):
        scores = self.score_proj(
            torch.tanh(
                self.key_proj(key) + self.query_proj(query) + self.bias
            )
        ).squeeze(-1)
        attention = self.dropout(masked_softmax(scores, valid_lens))
        return torch.bmm(attention.unsqueeze(1), value).squeeze(1)


class BaseSelfAttention(nn.Module):
    def __init__(self, d_hid=64):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(self, inputs, valid_lengths=None):
        attention = self.att_fc2(self.att_pool(self.att_fc1(inputs))).squeeze(-1)
        if valid_lengths is not None:
            positions = torch.arange(inputs.shape[1], device=inputs.device)
            mask = positions.unsqueeze(0) >= valid_lengths.to(inputs.device).unsqueeze(1)
            attention = attention.masked_fill(mask, -1e6)
        attention = torch.softmax(attention, dim=1)
        return (attention.unsqueeze(2) * inputs).sum(axis=1)


class FuseBaseSelfAttention(nn.Module):
    def __init__(self, d_hid=64, d_head=4):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)
        self.d_hid = d_hid
        self.d_head = d_head

    def forward(self, inputs, val_a=None, val_b=None, a_len=None):
        attention = self.att_fc2(self.att_pool(self.att_fc1(inputs)))
        attention = attention.transpose(1, 2)
        if val_a is not None:
            positions = torch.arange(inputs.shape[1], device=inputs.device)
            valid_audio = positions.unsqueeze(0) < val_a.to(inputs.device).unsqueeze(1)
            valid_video = (
                positions.unsqueeze(0) >= a_len
            ) & (
                positions.unsqueeze(0)
                < a_len + val_b.to(inputs.device).unsqueeze(1)
            )
            valid = (valid_audio | valid_video).unsqueeze(1)
            attention = attention.masked_fill(~valid, -1e5)
        attention = torch.softmax(attention, dim=2)
        output = torch.matmul(attention, inputs)
        return output.reshape(output.shape[0], self.d_head * self.d_hid)


class MMActionClassifier(nn.Module):
    """FedMM audio/video classifier used by the UCF101 pipeline."""

    def __init__(
        self,
        num_classes,
        audio_input_dim,
        video_input_dim,
        d_hid=128,
        n_filters=32,
        en_att=False,
        att_name="",
        d_head=6,
    ):
        super().__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        supported_attention = {"multihead", "additive", "base", "fuse_base"}
        if self.en_att and self.att_name not in supported_attention:
            raise ValueError(
                "attention must be one of "
                + ", ".join(sorted(supported_attention))
            )

        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )
        self.audio_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.video_rnn = nn.GRU(
            input_size=video_input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        if self.att_name == "multihead":
            self.audio_att = nn.MultiheadAttention(
                embed_dim=d_hid, num_heads=4, dropout=self.dropout_p
            )
            self.video_att = nn.MultiheadAttention(
                embed_dim=d_hid, num_heads=4, dropout=self.dropout_p
            )
        elif self.att_name == "additive":
            self.audio_att = AdditiveAttention(d_hid=d_hid, d_att=128)
            self.video_att = AdditiveAttention(d_hid=d_hid, d_att=128)
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(d_hid=d_hid)
            self.video_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(d_hid=d_hid, d_head=d_head)

        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes),
            )
        else:
            self.audio_proj = nn.Linear(d_hid, d_hid // 2)
            self.video_proj = nn.Linear(d_hid, d_hid // 2)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes),
            )

    @staticmethod
    def _run_rnn(inputs, lengths, rnn):
        if torch.any(lengths <= 0):
            return rnn(inputs)[0]
        packed = pack_padded_sequence(
            inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, _ = rnn(packed)
        return pad_packed_sequence(outputs, batch_first=True)[0]

    def forward(self, x_audio, x_video, len_a, len_v):
        x_audio = self.audio_conv(x_audio)
        len_a = len_a // 8
        x_audio = self._run_rnn(x_audio, len_a, self.audio_rnn)
        x_video = self._run_rnn(x_video, len_v, self.video_rnn)

        if not self.en_att:
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            x_mm = torch.cat((x_audio, x_video), dim=1)
        elif self.att_name == "multihead":
            x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
            x_video, _ = self.video_att(x_video, x_video, x_video)
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
        elif self.att_name == "additive":
            x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
            x_video = self.video_att(x_video, x_video, x_video, len_v)
        elif self.att_name == "base":
            x_audio = self.audio_att(x_audio, len_a)
            x_video = self.video_att(x_video, len_v)
        else:
            audio_max_length = x_audio.shape[1]
            x_mm = torch.cat((x_audio, x_video), dim=1)
            x_mm = self.fuse_att(
                x_mm,
                val_a=len_a,
                val_b=len_v,
                a_len=audio_max_length,
            )

        if self.en_att and self.att_name != "fuse_base":
            x_mm = torch.cat((x_audio, x_video), dim=1)
        predictions = self.classifier(x_mm)
        return predictions, x_mm
