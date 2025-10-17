import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import LR

class Rede(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.fc1 = nn.Linear(6, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.proj_skip = nn.Linear(128, 64)

        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.1)

        # --- DUELING HEAD (Value + Advantage) ---
        self.val = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3)   # ações: {-1,0,1}
        )

        # buffers simbióticos (mantidos)
        self.historico_recompensa = []
        self.ultima_media = None
        self.ciclos_estaveis = 0
        self.limiar_homeostase = 0.015

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x1 = self.act(self.norm1(self.fc1(x)))
        skip = self.proj_skip(x1)
        x2 = self.act(self.norm2(self.fc2(x1))) + 0.2 * skip
        if self.training:
            x2 = self.dropout(x2)

        v  = self.val(x2)                         # [B,1]
        a  = self.adv(x2)                         # [B,3]
        q  = v + (a - a.mean(dim=1, keepdim=True))# [B,3]
        return q

    # 🌿 poda simbiótica adaptada
    def aplicar_poda(self, limiar_base=0.002):
        total = sum(p.numel() for p in self.parameters())
        count_podadas = 0
        with torch.no_grad():
            for nome, param in self.named_parameters():
                if "weight" in nome:
                    abs_mean = param.abs().mean().item()
                    limiar = limiar_base * (1 + abs_mean * 5)
                    mascara = param.abs() > limiar
                    count_podadas += torch.numel(param) - mascara.sum().item()
                    param.mul_(mascara)
        taxa_poda = count_podadas / total
        return taxa_poda

    # 🧠 neurogênese simbiótica adaptada
    def regenerar_sinapses(self, taxa_poda):
        if taxa_poda > 0.15:
            with torch.no_grad():
                for nome, param in self.named_parameters():
                    if "weight" in nome:
                        variancia = torch.var(param)
                        mascara = torch.rand_like(param) < 0.03
                        novos_valores = torch.randn_like(param) * (variancia.sqrt() * 0.5)
                        param.add_(mascara.float() * novos_valores)

    # 🔁 homeostase simbiótica
    def verificar_homeostase(self, media_recompensa):
        if media_recompensa is None:
            return
        self.historico_recompensa.append(media_recompensa)
        if len(self.historico_recompensa) < 20:
            return
        media_atual = np.mean(self.historico_recompensa[-10:])
        if self.ultima_media is not None:
            variacao = abs(media_atual - self.ultima_media)
            if variacao < self.limiar_homeostase:
                self.ciclos_estaveis += 1
            else:
                self.ciclos_estaveis = 0
            if self.ciclos_estaveis >= 15:
                self.reciclar_simbiose()
                self.ciclos_estaveis = 0
        self.ultima_media = media_atual

    def reciclar_simbiose(self):
        with torch.no_grad():
            for nome, param in self.named_parameters():
                if "weight" in nome:
                    ruido = torch.randn_like(param) * 0.02
                    param.add_(ruido)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.reset_parameters()

def criar_modelo(device):
    modelo = Rede().to(device)
    alvo = Rede().to(device)
    alvo.load_state_dict(modelo.state_dict())

    loss_mse = nn.MSELoss()
    loss_smooth = nn.SmoothL1Loss(beta=0.8)
    def loss_fn(pred, alvo_t):
        return 0.7 * loss_mse(pred, alvo_t) + 0.3 * loss_smooth(pred, alvo_t)

    opt = optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
    return modelo, alvo, opt, loss_fn
