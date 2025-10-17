import time
import numpy as np
import threading
import pygame
import random
from config import *
from .flappy_physics import atualizar_movimento
from .flappy_render import renderizar_ambiente
from .flappy_utils import carregar_max_score
from visuals.fractal_bg import FractalBackground

from .flappy_audio import trilha_simbio_fluida

# ==========================================
# 🕊️ Ambiente Flappy EtherSym
# ==========================================
class AmbienteFlappy:
    """Flappy EtherSym — ambiente simbiótico com física, fractal e trilha harmônica viva."""

    def __init__(self):
        self.max_score_path = "max_score.json"
        self.max_score = carregar_max_score(self.max_score_path)
        self.reset()
        self.last_render = 0.0
        self.imunidade_ativa = False
        self.imunidade_fim = 0.0
        self.cano_colidido = None
        self.bg = FractalBackground(LARGURA, ALTURA)

        # === 🎶 Inicia trilha simbiótica infinita ===
        # 🎶 inicia a trilha sonora simbiótica fluida em thread separada
        threading.Thread(
            target=trilha_simbio_fluida,
            args=(lambda: float(self.passaro["energia"]),),
            daemon=True
        ).start()


    # ========================
    # Reset do ambiente
    # ========================
    def reset(self):
        self.passaro = {"x": 60, "y": ALTURA // 2, "vel": 0.0, "energia": 1.0}
        self.canos = [{"x": 300, "altura": 300, "scored": False}]
        self.pontuacao = 0
        self.vivo = True
        self.gravidade_base = GRAVIDADE_BASE
        self.imunidade_ativa = False
        self.imunidade_fim = 0.0
        self.cano_colidido = None
        return self._get_estado(0)

    # ========================
    # Estado simbiótico
    # ========================
    def _get_estado(self, acao_prev):
        p = self.passaro
        canos_validos = [c for c in self.canos if c["x"] > p["x"] - 50]
        cano_proximo = min(canos_validos, key=lambda c: c["x"], default=self.canos[0])
        dist_x = ((cano_proximo["x"] - p["x"]) / (LARGURA / 2)) - 1.0
        dist_y = np.clip((cano_proximo["altura"] - p["y"]) / (ALTURA / 2), -1.0, 1.0)
        vel = np.tanh(p["vel"] / 10.0)
        alt = (p["y"] / ALTURA) * 2 - 1
        energia = (p["energia"] * 2) - 1
        return np.array([dist_x, dist_y, vel, alt, energia, acao_prev], dtype=np.float32)

    # ========================
    # Passo de simulação
    # ========================
    def step(self, acao, campo):
        if not self.vivo:
            return self._get_estado(acao), -100.0, True
        self.cano_colidido = None
        novo_estado, recompensa, cano_colidido, vivo = atualizar_movimento(self, acao, campo)
        self.vivo = vivo
        self.cano_colidido = cano_colidido
        return novo_estado, recompensa, not self.vivo

    # ========================
    # Renderização (sem flicker)
    # ========================
    def render(self, campo):
        renderizar_ambiente(self, campo)
