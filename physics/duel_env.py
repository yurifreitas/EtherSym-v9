# ==========================================
# 🌌 EtherSym v9.9 — Ambiente de Duelo Simbiótico (fix)
# ==========================================
import numpy as np
import random, copy
from config import ALTURA, CHAO, VELOCIDADE_PULO
from physics.flappy_env import AmbienteFlappy

class BirdState:
    __slots__ = ("y", "vy", "alive", "score")
    def __init__(self, y: float, vy: float = 0.0):
        self.y = float(y)
        self.vy = float(vy)
        self.alive = True
        self.score = 0.0

class DuelEnv(AmbienteFlappy):
    """Dois agentes (Humano/IA) no mesmo campo compartilhado."""
    def __init__(self):
        super().__init__()
        # tenta spawnar no y oficial do seu env; fallback = centro
        y0 = getattr(self, "passaro_y", ALTURA // 2)
        self.human = BirdState(y0)
        self.ai    = BirdState(y0)
        self.t = 0
        # caches seguros
        self._x_ref = getattr(self, "passaro_x", 80)
        self._vel_cano = float(getattr(self, "velocidade_cano", 5.0))

    # ------------------------------------------
    # 🧠 Clonagem de estado
    # ------------------------------------------
    def get_state(self):
        return {
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "pipes": copy.deepcopy(getattr(self, "pipes", [])),
            "t": self.t,
        }

    def set_state(self, st):
        random.setstate(st["rng_py"])
        np.random.set_state(st["rng_np"])
        self.pipes = copy.deepcopy(st["pipes"])
        self.t = int(st["t"])

    # ------------------------------------------
    # 🌍 Atualização do mundo (canos/gravity clock)
    # ------------------------------------------
    def _ensure_pipes(self):
        """Garante que exista ao menos um conjunto de canos no início."""
        if not hasattr(self, "pipes"):
            self.pipes = []
        if len(self.pipes) == 0 and hasattr(self, "spawn_pipe"):
            # usa método do seu ambiente, se existir
            try:
                self.spawn_pipe()
            except Exception:
                # fallback neutro: 1 pipe a frente com gap central
                self.pipes = [{"x": self._x_ref + 200, "gap_y": ALTURA * 0.5, "gap": 220}]

    def advance_world(self, campo):
        """Avança obstáculos sem tocar nos pássaros."""
        self.t += 1
        self._ensure_pipes()

        # se o seu AmbienteFlappy tiver um método dedicado para pipes, use-o
        if hasattr(self, "update_pipes"):
            self.update_pipes()
        else:
            # fallback mínimo: mover e reciclar pipes manualmente
            vel = float(getattr(self, "velocidade_cano", self._vel_cano))
            for p in list(self.pipes):
                p["x"] -= vel
            # remove os que saíram e cria novos
            self.pipes = [p for p in self.pipes if p["x"] > -50]
            if len(self.pipes) < 1 or (self.pipes and self.pipes[-1]["x"] < self._x_ref + 120):
                # cria novo pipe simples
                gap = 220
                gap_y = np.clip(np.random.normal(loc=ALTURA*0.5, scale=ALTURA*0.15), 120, CHAO-120)
                self.pipes.append({"x": self._x_ref + 250, "gap_y": float(gap_y), "gap": float(gap)})

    # ------------------------------------------
    # 🪶 Física do pássaro
    # ------------------------------------------
    def _apply_action(self, bird, acao):
        """Ação: 1 = pulo; 0 = neutro; -1 = leve descida."""
        if acao == 1:
            bird.vy = VELOCIDADE_PULO * 1.10   # pulo um pouco mais forte
        elif acao == -1:
            bird.vy += 0.20                    # descida suave

    def _integrate_bird(self, bird, campo):
        """Gravidade estável + clamp de velocidade."""
        # pequeno ‘warmup’ para não morrer no spawn
        if self.t < 10:
            return
        g = campo.gravity(self.t) if hasattr(campo, "gravity") else 1.5
        bird.vy += g * 0.15
        # clamp para evitar queda infinita
        if bird.vy > 12.0: bird.vy = 12.0
        if bird.vy < -10.0: bird.vy = -10.0
        bird.y += bird.vy

    # ------------------------------------------
    # 🧱 Colisão + recompensa
    # ------------------------------------------
    def _collide_and_reward(self, bird):
        reward = 0.5  # ficar vivo vale algo
        # tolerância de borda (evita morte instantânea no spawn)
        if bird.y < 8 or bird.y > CHAO - 8:
            return -10.0, True

        # colisão com canos (fallback simples)
        hit = False
        if hasattr(self, "pipes") and self.pipes:
            x_bird = self._x_ref
            for p in self.pipes:
                dx = abs(p["x"] - x_bird)
                # largura de colisão simples ~ 25 px
                if dx < 25:
                    gap = float(p.get("gap", 220.0))
                    gap_y = float(p.get("gap_y", ALTURA*0.5))
                    if not (gap_y - gap/2.0 < bird.y < gap_y + gap/2.0):
                        hit = True
                        break
            # shaping: aproximar do gap_y dá bônus leve
            lead = self.pipes[0]
            reward += 0.003 * (220.0 - min(220.0, abs(lead["gap_y"] - bird.y)))

        if hit:
            return -10.0, True

        # pontuação incremental
        return reward, False

    # ------------------------------------------
    # 👁️ Observação (precisa bater com o treino)
    # Esperado pelo modelo: 6 inputs [y, vy, dx, dy, gap, vel_cano]
    # ------------------------------------------
    def _encode_obs(self, bird):
        dx = dy = gap = vel_cano = 0.0
        if hasattr(self, "pipes") and self.pipes:
            pipe = self.pipes[0]
            dx = float(pipe["x"] - self._x_ref)
            dy = float(pipe["gap_y"] - bird.y)
            gap = float(pipe.get("gap", 220.0))
        vel_cano = float(getattr(self, "velocidade_cano", self._vel_cano))
        return np.array([bird.y, bird.vy, dx, dy, gap, vel_cano], dtype=np.float32)

    # ------------------------------------------
    # ⚔️ Passo de duelo
    # ------------------------------------------
    def step_duel(self, acao_humano, acao_ia, campo):
        self.advance_world(campo)

        # humano
        if self.human.alive:
            self._apply_action(self.human, acao_humano)
            self._integrate_bird(self.human, campo)
            r_h, d_h = self._collide_and_reward(self.human)
            if d_h: self.human.alive = False
        else:
            r_h, d_h = 0.0, True

        # IA
        if self.ai.alive:
            self._apply_action(self.ai, acao_ia)
            self._integrate_bird(self.ai, campo)
            r_ai, d_ai = self._collide_and_reward(self.ai)
            if d_ai: self.ai.alive = False
        else:
            r_ai, d_ai = 0.0, True

        self.human.score += r_h
        self.ai.score += r_ai

        return (
            (self._encode_obs(self.human), self._encode_obs(self.ai)),
            (r_h, r_ai),
            (d_h, d_ai),
        )

    # ------------------------------------------
    # ♻️ Reinício
    # ------------------------------------------
    def reset_duel(self):
        # reseta mundo original
        self.reset()
        # re-lê caches seguros
        self._x_ref = getattr(self, "passaro_x", 80)
        self._vel_cano = float(getattr(self, "velocidade_cano", 5.0))
        # reseta agentes
        y0 = getattr(self, "passaro_y", ALTURA // 2)
        self.human = BirdState(y0)
        self.ai    = BirdState(y0)
        self.t = 0
        # garante ao menos um pipe inicial
        self._ensure_pipes()
