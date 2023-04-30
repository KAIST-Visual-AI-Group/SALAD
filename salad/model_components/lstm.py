import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, text_dim, embedding_dim, vocab_size, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(embedding_dim, text_dim, batch_first=True)
        self.w_attn = nn.Parameter(torch.Tensor(1, text_dim))
        nn.init.xavier_uniform_(self.w_attn)

    def forward(self, padded_tokens, dropout=0.5):
        w_emb = self.word_embedding(padded_tokens)
        w_emb = F.dropout(w_emb, dropout, self.training)
        len_seq = (padded_tokens != self.padding_idx).sum(dim=1).cpu()
        x_packed = pack_padded_sequence(
            w_emb, len_seq, enforce_sorted=False, batch_first=True
            )
        B = padded_tokens.shape[0]
        rnn_out, _ = self.rnn(x_packed)
        rnn_out, dummy = pad_packed_sequence(rnn_out, batch_first=True)
        h = rnn_out[torch.arange(B), len_seq - 1]
        final_feat, attn = self.word_attention(rnn_out, h, len_seq)
        return final_feat, attn

    def word_attention(self, R, h, len_seq):
        """
        Input:
            R: hidden states of the entire words
            h: the final hidden state after processing the entire words
            len_seq: the length of the sequence
        Output:
            final_feat: the final feature after the bilinear attention
            attn: word attention weights
        """
        B, N, D = R.shape
        device = R.device
        len_seq = len_seq.to(device)

        W_attn = (self.w_attn * torch.eye(D).to(device))[None].repeat(B, 1, 1)
        score = torch.bmm(torch.bmm(R, W_attn), h.unsqueeze(-1))

        mask = torch.arange(N).reshape(1, N, 1).repeat(B, 1, 1).to(device)
        mask = mask < len_seq.reshape(B, 1, 1)

        score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, 1)
        final_feat = torch.bmm(R.transpose(1, 2), attn).squeeze(-1)

        return final_feat, attn.squeeze(-1)
