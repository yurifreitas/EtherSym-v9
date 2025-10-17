import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import *

class Rede(nn.Module):
    def __init__(self):
        super().__init__()

        # === estrutura base ===
        self.fc1 = nn.Linear(6, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.proj_skip = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.1)

        # buffer simbiótico de recompensas
        self.historico_recompensa = []
        self.ultima_media = None
        self.ciclos_estaveis = 0
        self.limiar_homeostase = 0.015  # variação mínima pra considerar estagnação

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
        return self.fc3(x2)

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
        print(f"🌿 Poda simbiótica adaptada: {taxa_poda*100:.2f}% dos pesos removidos")
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
            print("🧠 Neurogênese simbiótica: conexões regeneradas")

    # 🔁 homeostase simbiótica (auto-reciclagem)
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

            # 🔄 se ficou estável por muito tempo, recicla
            if self.ciclos_estaveis >= 15:
                self.reciclar_simbiose()
                self.ciclos_estaveis = 0
        self.ultima_media = media_atual

    def reciclar_simbiose(self):
        print("\n♻️  Reciclagem simbiótica iniciada — a rede está se renovando...\n")
        with torch.no_grad():
            for nome, param in self.named_parameters():
                if "weight" in nome:
                    ruido = torch.randn_like(param) * 0.02
                    param.add_(ruido)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.reset_parameters()
        print("✨ Rede simbiótica regenerada e estabilizada!\n")

# ========================
# Criação do modelo simbiótico
# ========================

def criar_modelo(device):
    modelo = Rede().to(device)
    alvo = Rede().to(device)
    alvo.load_state_dict(modelo.state_dict())

    loss_mse = nn.MSELoss()
    loss_smooth = nn.SmoothL1Loss(beta=0.8)
    def loss_fn(pred, alvo):
        return 0.7 * loss_mse(pred, alvo) + 0.3 * loss_smooth(pred, alvo)

    otimizador = optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
    print("🧬 Modelo simbiótico criado com homeostase adaptativa e regeneração automática.")
    return modelo, alvo, otimizador, loss_fn
