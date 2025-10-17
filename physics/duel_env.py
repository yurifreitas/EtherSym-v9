# ==========================================
# 🪶 DuelEnv — Duelo Humano vs IA (EtherSym)
# ==========================================
import numpy as np
from physics.flappy_env import AmbienteFlappy


class DuelEnv:
    """Ambiente simbiótico de duelo entre Humano e IA — compartilham o mesmo campo."""

    def __init__(self):
        # Mundo simbiótico compartilhado
        self.world = AmbienteFlappy()
        self.world.reset()

        # Parâmetros físicos básicos
        self.chao = getattr(self.world, "CHAO", 520)
        self.largura = getattr(self.world, "LARGURA", 400)
        self.altura = getattr(self.world, "ALTURA", 600)

        # Pássaros inicializados com offsets verticais
        self.human = self._criar_bird(offset_y=-60)
        self.ai = self._criar_bird(offset_y=60)

    # --------------------------------------
    def _criar_bird(self, offset_y=0):
        """Cria um pássaro simbiótico baseado no estado atual do mundo."""
        p = self.world.passaro
        return {
            "y": p["y"] + offset_y,
            "vel": 0.0,
            "alive": True,
            "score": 0
        }

    # --------------------------------------
    def _prox_cano(self):
        """Seleciona o cano mais próximo do pássaro principal."""
        p = self.world.passaro
        canos = [c for c in self.world.canos if c["x"] + 80 > p["x"] - 50]
        return min(canos, key=lambda c: c["x"], default=self.world.canos[0])

    # --------------------------------------
    def _encode_obs(self, bird, acao_prev=0):
        """Codifica o estado simbiótico (6 entradas compatíveis com o modelo treinado)."""
        cano = self._prox_cano()
        p_ref = self.world.passaro

        dist_x = ((cano["x"] - p_ref["x"]) / (self.largura / 2)) - 1.0
        dist_y = np.clip((cano["altura"] - bird["y"]) / (self.altura / 2), -1.0, 1.0)
        vel = np.tanh(bird["vel"] / 10.0)
        alt = (bird["y"] / self.altura) * 2 - 1.0
        energia = getattr(p_ref, "energia", 1.0) * 2 - 1.0
        return np.array([dist_x, dist_y, vel, alt, energia, acao_prev], dtype=np.float32)

    # --------------------------------------
    def _apply_action(self, bird, action):
        """Aplica a ação simbiótica do jogador."""
        if not bird["alive"]:
            return
        if action == 1:  # pulo
            bird["vel"] = -9.0
        elif action == -1:  # descida rápida
            bird["vel"] += 3.0

    # --------------------------------------
    def _physics_step(self, bird, campo):
        """Simula física e colisão simplificada."""
        if not bird["alive"]:
            return

        g = getattr(campo, "gravidade", 1.8)
        bird["vel"] += g * 0.4
        bird["y"] += bird["vel"]

        # chão/teto
        if bird["y"] >= self.chao or bird["y"] <= 0:
            bird["alive"] = False
            return

        # colisão com cano
        cano = self._prox_cano()
        gap_top = cano["altura"] - 100
        gap_bottom = cano["altura"] + 100
        if (cano["x"] < 80 < cano["x"] + 80) and (bird["y"] < gap_top or bird["y"] > gap_bottom):
            bird["alive"] = False

        # pontuação por cano
        if not cano.get("scored") and cano["x"] + 80 < 80:
            bird["score"] += 1
            cano["scored"] = True

    # --------------------------------------
    def step_duel(self, a_h, a_ai, campo):
        """Executa um passo simbiótico de duelo entre Humano e IA."""
        self._apply_action(self.human, a_h)
        self._apply_action(self.ai, a_ai)

        # Atualiza o campo e obstáculos do mundo compartilhado
        self.world.step(0, campo)

        # Física independente dos pássaros
        self._physics_step(self.human, campo)
        self._physics_step(self.ai, campo)

        # Recompensas
        r_h = 1.0 if self.human["alive"] else -5.0
        r_ai = 1.0 if self.ai["alive"] else -5.0

        # Condição de fim
        d_h = not self.human["alive"]
        d_ai = not self.ai["alive"]

        # Observações retornadas (6 entradas cada)
        return (
            (self._encode_obs(self.human, a_h), self._encode_obs(self.ai, a_ai)),
            (r_h, r_ai),
            (d_h, d_ai)
        )

    # --------------------------------------
    def render(self, campo):
        """Renderiza o mundo físico."""
        self.world.render(campo)

    # --------------------------------------
    def reset_duel(self):
        """Reinicia o duelo completo."""
        self.world.reset()
        self.human = self._criar_bird(offset_y=-60)
        self.ai = self._criar_bird(offset_y=60)
        return (self._encode_obs(self.human), self._encode_obs(self.ai))
