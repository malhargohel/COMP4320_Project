import torch
import torch.nn as nn

class TransformerAgent(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, dropout=0.1):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(state_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=2
        )
        self.actor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim)
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, state_dim)
        x = self.embedding(x)
        x = self.norm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        action_logits = self.actor(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value