import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super(TransformerFusion, self).__init__()
        # self.vision_dim = vision_dim
        # self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Vision and text embedding layers
        # self.vision_embedding = nn.Linear(vision_dim, hidden_dim)
        # self.text_embedding = nn.Linear(text_dim, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_feats, text_feats):
        # Embed vision and text features
        # embedded_vision = self.vision_embedding(vision_feats)  # [batch_size, vision_dim] -> [batch_size, hidden_dim]
        # embedded_text = self.text_embedding(text_feats)  # [batch_size, text_dim] -> [batch_size, hidden_dim]

        # Concatenate the vision and text embeddings
        fused_feats = torch.cat((vision_feats, text_feats), dim=0)  # [2*batch_size, hidden_dim]

        # Transpose the fused features to [sequence_length, batch_size, hidden_dim]
        fused_feats = fused_feats.transpose(0, 1).unsqueeze(1)  # [sequence_length, 1, batch_size, hidden_dim]

        # Pass the fused features through the Transformer encoder
        fused_feats = self.transformer_encoder(fused_feats)  # [sequence_length, 1, batch_size, hidden_dim]

        # Squeeze the fused features to [batch_size, hidden_dim]
        fused_feats = fused_feats.squeeze().transpose(0, 1)  # [1, batch_size, hidden_dim] -> [batch_size, hidden_dim]

        # Pass the fused features through an output layer
        fused_feats = self.output_layer(fused_feats)  # [batch_size, hidden_dim] -> [batch_size, hidden_dim]

        return fused_feats