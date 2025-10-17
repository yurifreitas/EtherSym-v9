import time
import numpy as np
import threading
import pygame
import random
from config import *
from .flappy_physics import atualizar_movimento
from .flappy_render import renderizar_ambiente
from .flappy_utils import carregar_max_score, salvar_max_score
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

        # === 🎶 Trilha simbiótica em thread ===
        threading.Thread(
            target=trilha_simbio_fluida,
            args=(lambda: float(self.passaro["energia"]),),
            daemon=True
        ).start()

    # ========================
    # Reset do ambiente
    # ========================
    def reset(self):
        # Estado inicial do pássaro
        self.passaro = {"x": 80.0, "y": ALTURA // 2, "vel": 0.0, "energia": 1.0}

        # Inicialização de canos e pontuação
        self.canos = [{"x": 400, "altura": ALTURA // 2, "scored": False}]
        self.pontuacao = 0
        self.vivo = True
        self.gravidade_base = GRAVIDADE_BASE
        self.imunidade_ativa = False
        self.imunidade_fim = 0.0
        self.cano_colidido = None

        # Linha fixa de pontuação no centro da tela (posição visual do pássaro)
        self.score_x_ref = 80.0
        self.cano_larg = 70.0

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

        # --- Atualiza pontuação estável ---
        linha_score = self.score_x_ref
        cano_larg = self.cano_larg
        for cano in self.canos:
            if "scored" not in cano:
                cano["scored"] = False
            # ✅ Conta quando a borda direita do cano cruza a linha de pontuação
            if (not cano["scored"]) and (cano["x"] + cano_larg) < linha_score:
                cano["scored"] = True
                self.pontuacao += 1
                self.max_score = max(self.max_score, self.pontuacao)
                salvar_max_score(self.max_score_path, self.max_score)
                recompensa += 25.0 * random.uniform(0.8, 1.2)
                # 🔊 Log visual para depuração
                print(f"🎯 +1 ponto! score={self.pontuacao}")

        self.vivo = vivo
        self.cano_colidido = cano_colidido
        return novo_estado, recompensa, not self.vivo

    # ========================
    # Renderização
    # ========================
    def render(self, campo):
        renderizar_ambiente(self, campo)
