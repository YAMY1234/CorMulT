import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a position encoding matrix of size [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Use sin for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Use cos for odd dimensions
        pe = pe.unsqueeze(0)  # Add batch dimension [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]
        Add positional encoding to x
        """
        length = x.size(1)
        x = x + self.pe[:, :length, :]
        return self.dropout(x)


class CorrelationModel(nn.Module):
    def __init__(self, 
                 text_in_dim=300, 
                 audio_in_dim=74, 
                 vision_in_dim=35, 
                 d_model=128,
                 num_layers=3,
                 num_heads=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 out_dim=64):
        """
        Parameter description:
        text_in_dim: The dimension of text input features (default 300)
        audio_in_dim: The dimension of audio input features (default 74)
        vision_in_dim: The dimension of vision input features (default 35)
        d_model: Transformer hidden layer dimension
        num_layers: Number of Transformer layers
        num_heads: Number of heads in MultiheadAttention
        dim_feedforward: The dimension of the intermediate layer in the FFN
        dropout: The dropout rate
        out_dim: The dimension of the mapped shared space
        """
        super(CorrelationModel, self).__init__()
        
        # Map to a unified d_model dimension
        self.text_fc = nn.Linear(text_in_dim, d_model)
        self.audio_fc = nn.Linear(audio_in_dim, d_model)
        self.vision_fc = nn.Linear(vision_in_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define the TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   batch_first=True)
        # Create the TransformerEncoder
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.vision_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Finally, map to the shared space dimension out_dim
        self.text_out_fc = nn.Linear(d_model, out_dim)
        self.audio_out_fc = nn.Linear(d_model, out_dim)
        self.vision_out_fc = nn.Linear(d_model, out_dim)

    def forward(self, text, audio, vision):
        # text: [B, T_l, 300]
        # audio: [B, T_a, 74]
        # vision: [B, T_v, 35]

        # Linear mapping to d_model dimension
        t_emb = self.text_fc(text)    # [B, T_l, d_model]
        a_emb = self.audio_fc(audio)  # [B, T_a, d_model]
        v_emb = self.vision_fc(vision)# [B, T_v, d_model]

        # Add positional encoding
        t_emb = self.pos_encoder(t_emb)
        a_emb = self.pos_encoder(a_emb)
        v_emb = self.pos_encoder(v_emb)

        # Pass through TransformerEncoders
        t_enc = self.text_encoder(t_emb)   # [B, T_l, d_model]
        a_enc = self.audio_encoder(a_emb)  # [B, T_a, d_model]
        v_enc = self.vision_encoder(v_emb) # [B, T_v, d_model]

        # Map to shared space
        t_out = self.text_out_fc(t_enc)    # [B, T_l, out_dim]
        a_out = self.audio_out_fc(a_enc)   # [B, T_a, out_dim]
        v_out = self.vision_out_fc(v_enc)  # [B, T_v, out_dim]

        return t_out, a_out, v_out
