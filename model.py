import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix a shape(seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)
        # Create a vector of shape(seq_len, q)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        positional_encoding[:, 1::2] = torch.sin(position * div_term)

        # Apply a new dimension to handle batch of sentences 
        positional_encoding = positional_encoding.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False);
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    # For numerical stability and to avoid the division by zero, Epsilon(eps) is introduced in normalization layer
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert d_model % head == 0, "d_model is not divisble by head"

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # @ is matrix multiplication in python
        # (batch, head, seq_len, d_k) * (batch, head, d_k, seq_len) --> (batch, head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, head, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, head, seq_len, seq_len) * (batch, head, seq_len, d_k) --> (batch, head, seq_len, d_k) and (batch, head, seq_len, seq_len)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) * (Batch, d_model, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) * (Batch, d_model, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) * (Batch, d_model, d_model) --> (Batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, head, d_k) --> (Batch, head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1],  self.head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],  self.head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1],  self.head, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Transpose(batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head * self.d_k)

        # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_layer = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, targt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, targt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_layer)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, targt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, targt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, targt_embedding: InputEmbeddings, src_pos: PositionalEncoding, targt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.targt_embedding = targt_embedding
        self.src_pos = src_pos
        self.targt_pos = targt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, targt, targt_mask):
        targt = self.targt_embedding(targt)
        targt = self.targt_pos(targt)
        return self.decoder(targt, encoder_output, src_mask, targt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, targt_vocab_size: int, src_seq_len: int, targt_seq_len: int, d_model: int = 512, N: int = 6, head: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create Input Embeddings layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    targt_embed = InputEmbeddings(d_model, targt_vocab_size)

    # Create Positional Embeddings layers
    src_pos_embed = PositionalEncoding(d_model, src_seq_len, dropout)
    targt_pos_embed = PositionalEncoding(d_model, targt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the Projection Layer
    projection_layer = ProjectionLayer(d_model, targt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, targt_embed, src_pos_embed, targt_pos_embed, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer