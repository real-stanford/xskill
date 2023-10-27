from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from xskill.model.network import create_mlp


class TorchTransformerEncoder(nn.Module):

    def __init__(self,
                 query_dim,
                 heads,
                 dim_feedforward=2048,
                 n_layer=1,
                 rep_dim=None,
                 use_encoder=False,
                 input_dim=None,
                 pos_encoder=None,
                 repeat_input=False) -> None:
        super().__init__()

        self.use_encoder = use_encoder
        self.repeat_input = repeat_input
        if self.use_encoder:
            self.encoder = nn.Linear(input_dim, query_dim)
        else:
            self.encoder = None

        encode_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.pos_encoder = pos_encoder
        self.d_model = query_dim
        self.transformer_encoder = nn.TransformerEncoder(encode_layer,
                                                         num_layers=n_layer)
        self.decoder = nn.Linear(query_dim, rep_dim)

        representation_token = nn.Parameter(torch.randn(query_dim))
        self.register_parameter("representation_token", representation_token)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if self.use_encoder:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        batch, _, f = x.shape
        if self.repeat_input:
            x = x.repeat(1, 1, int(self.d_model // f))
        if self.use_encoder:
            x = self.encoder(x)

        x = torch.cat([
            self.representation_token.unsqueeze(0).expand(batch,
                                                          -1).unsqueeze(1), x
        ],
                      dim=1)
        x = x * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)

        output = self.transformer_encoder(x)
        output = output[:, 0]
        output = self.decoder(output)
        return output


class TorchTransformerProtoPredictor(nn.Module):

    def __init__(self,
                 query_dim,
                 heads,
                 dim_feedforward=2048,
                 n_layer=1,
                 proto_dim=None,
                 use_encoder=False,
                 input_dim=None,
                 pos_encoder=None) -> None:
        super().__init__()

        self.use_encoder = use_encoder
        if self.use_encoder:
            self.encoder = nn.Linear(input_dim, query_dim)
        else:
            self.encoder = None

        encode_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.pos_encoder = pos_encoder
        self.d_model = query_dim
        self.transformer_encoder = nn.TransformerEncoder(encode_layer,
                                                         num_layers=n_layer)
        self.decoder = nn.Linear(query_dim, proto_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if self.use_encoder:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, proto_snap):
        batch, _, _ = proto_snap.shape
        if self.use_encoder:
            x = self.encoder(x)  #(B,query_dim)
            x = x.unsqueeze(1)  #(B,1,query_dim)

        x = torch.cat([x, proto_snap], dim=1)  #(B,1+T,query_dim)
        x = x * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)

        output = self.transformer_encoder(x)
        output = output[:, 0]
        output = self.decoder(output)
        return output


class TorchTransformerProtoCls(nn.Module):

    def __init__(self,
                 query_dim,
                 heads,
                 dim_feedforward=2048,
                 n_layer=1,
                 use_encoder=False,
                 input_dim=None,
                 pos_encoder=None) -> None:
        super().__init__()

        self.use_encoder = use_encoder
        if self.use_encoder:
            self.encoder = nn.Linear(input_dim, query_dim)
        else:
            self.encoder = None

        encode_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.pos_encoder = pos_encoder
        self.d_model = query_dim
        self.transformer_encoder = nn.TransformerEncoder(encode_layer,
                                                         num_layers=n_layer)
        self.decoder = nn.Linear(query_dim, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if self.use_encoder:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, proto_snap):
        batch, _, _ = proto_snap.shape
        if self.use_encoder:
            x = self.encoder(x)  #(B,query_dim)
            x = x.unsqueeze(1)  #(B,1,query_dim)

        x = torch.cat([x, proto_snap], dim=1)  #(B,1+T,query_dim)
        x = x * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = x + self.pos_encoder(x)

        output = self.transformer_encoder(x)  #(B,1+T,query_dim)
        output = output[:, 1:]
        output = self.decoder(output)  # proto energey
        return output.squeeze(-1)  #(B,T)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val):
    return val is not None


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 pe,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 rep_dim=32,
                 to_out_dim=None,
                 net_arch=[128, 128],
                 use_layer_norm=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.pe = pe
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.use_layer_norm = use_layer_norm

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, default(to_out_dim, query_dim)), )

        if self.use_layer_norm:
            self.ln = nn.LayerNorm(default(to_out_dim, query_dim))

        head = create_mlp(
            input_dim=default(to_out_dim, query_dim),
            output_dim=rep_dim,
            net_arch=net_arch,
        )
        self.head = nn.Sequential(*head)

        representation_token = nn.Parameter(torch.randn(query_dim))
        self.register_parameter("representation_token", representation_token)

    def forward(self, x, context=None, mask=None):
        batch, _, _ = x.shape
        x = torch.cat([
            self.representation_token.unsqueeze(0).expand(batch,
                                                          -1).unsqueeze(1), x
        ],
                      dim=1)

        if self.pe is not None:
            x = x + self.pe(x)

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        # take out the representation token
        out = out[:, 0]
        if self.use_layer_norm:
            out = self.ln(out)
        out = self.head(out)
        return out


class Attention(nn.Module):

    def __init__(self,
                 query_dim,
                 heads=8,
                 dim_head=64,
                 attn_pdrop=0,
                 resid_pdrop=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(inner_dim, query_dim)

    def forward(self, x):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # output projection
        out = self.resid_drop(self.proj(out))
        return out


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self,
                 query_dim,
                 heads=8,
                 dim_head=64,
                 attn_pdrop=0,
                 resid_pdrop=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.attn = Attention(query_dim=query_dim,
                              heads=heads,
                              dim_head=dim_head,
                              attn_pdrop=attn_pdrop,
                              resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, 4 * query_dim),
            nn.GELU(),
            nn.Linear(4 * query_dim, query_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self,
                 query_dim,
                 heads=8,
                 dim_head=64,
                 attn_pdrop=0,
                 resid_pdrop=0,
                 embd_pdrop=0,
                 n_layer=1,
                 final_output_dim=None,
                 pe=None) -> None:
        super().__init__()
        representation_token = nn.Parameter(torch.randn(query_dim))
        self.register_parameter("representation_token", representation_token)
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[
            Block(query_dim, heads, dim_head, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])
        # decoder head
        self.ln_f = nn.LayerNorm(query_dim)
        self.head = nn.Linear(query_dim, final_output_dim, bias=False)
        self.apply(self._init_weights)

        self.pe = pe

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        batch, _, _ = x.shape
        x = torch.cat([
            self.representation_token.unsqueeze(0).expand(batch,
                                                          -1).unsqueeze(1), x
        ],
                      dim=1)
        if self.pe is not None:
            x = x + self.pe(x)

        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        representation = x[:, 0]
        representation = self.head(representation)
        return representation


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the input for as many time
    steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self,
                 size: int = 0,
                 max_len: int = 5000,
                 frequency=10000.0) -> None:
        """
        Positional Encoding with maximum length
        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        if size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={size})"
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(frequency) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, size)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """
        Embed inputs.
        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (seq_len, batch_size, dim)
        :return: positionally encoded word embeddings
        """
        # get position encodings
        return self.pe[:, :emb.size(1)]


class BBoxPositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    """

    def __init__(self, size: int = 0, frequency=10000.0) -> None:
        """
        Positional Encoding with maximum length
        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        if size % 4 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={size})"
            )
        # pe = torch.zeros(max_len, size)
        # position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 4, dtype=torch.float) *
                              -(math.log(frequency) / size)))
        super().__init__()
        self.register_buffer("div_term", div_term)
        self.dim = size

    def forward(self, bbox):
        """
        Embed inputs.
        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (batch_size, 4)
        :return: positionally encoded word embeddings
        """
        # get position encodings
        b, _ = bbox.shape
        pe = torch.zeros((b, self.dim), device=self.div_term.device)
        pe[:, 0::4] = torch.sin(bbox[:, 0].unsqueeze(-1) * self.div_term)
        pe[:, 1::4] = torch.cos(bbox[:, 1].unsqueeze(-1) * self.div_term)
        pe[:, 2::4] = torch.sin(bbox[:, 2].unsqueeze(-1) * self.div_term)
        pe[:, 3::4] = torch.cos(bbox[:, 3].unsqueeze(-1) * self.div_term)

        return pe


class BBoxLearnedEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    """

    def __init__(self, size: int = 0, scaling=112) -> None:
        """
        Positional Encoding with maximum length
        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        super().__init__()
        self.size = size
        self.scaling = scaling
        self.pe = nn.Sequential(nn.Linear(4, self.size))

    def forward(self, bbox):
        """
        Embed inputs.
        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (batch_size, 4)
        :return: positionally encoded word embeddings
        """
        pe = self.pe(bbox / self.scaling)
        return pe


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x