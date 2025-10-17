import pygame, random, numpy as np, math, time, os, json
from config import *

class AmbienteFlappy:
    """Flappy EtherSym — física simbiótica contínua com detecção visual de colisão."""

    def __init__(self):
        self.max_score_path = "max_score.json"
        self.max_score = self._carregar_max_score()
        self.reset()
        self.last_render = 0.0
        self.imunidade_ativa = False
        self.imunidade_fim = 0.0
        self.cano_seguro = None
        self.cano_colidido = None  # ← marca qual cano foi atingido

    # ========================
    # Recorde simbiótico
    # ========================
    def _carregar_max_score(self):
        if os.path.exists(self.max_score_path):
            try:
                with open(self.max_score_path, "r") as f:
                    return json.load(f).get("max_score", 0)
            except Exception:
                return 0
        return 0

    def _salvar_max_score(self):
        try:
            with open(self.max_score_path, "w") as f:
                json.dump({"max_score": self.max_score}, f)
        except Exception:
            pass

    # ========================
    # Reinício
    # ========================
    def reset(self):
        self.passaro = {"x": 60, "y": ALTURA // 2, "vel": 0.0, "energia": 1.0}
        self.canos = [{"x": 300, "altura": random.randint(180, 420), "scored": False}]
        self.pontuacao = 0
        self.vivo = True
        self.gravidade_base = GRAVIDADE_BASE
        self.imunidade_ativa = False
        self.imunidade_fim = 0.0
        self.cano_seguro = None
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
    # Passo simbiótico contínuo
    # ========================
    def step(self, acao, campo):
        if not self.vivo:
            return self._get_estado(acao), -100.0, True  # morto de fato

        p = self.passaro
        recompensa = 0.0
        agora = time.time()
        self.cano_colidido = None  # reseta marca visual

        # Gravidade simbiótica
        g_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
        osc = math.sin(agora * 1.6 + random.random()) * OSCILACAO_AMPLITUDE
        gravidade = (self.gravidade_base * math.copysign(1, g_local)) + (g_local * 2.0) + osc
        p["vel"] += gravidade
        p["vel"] = float(np.clip(p["vel"], -12.0, 12.0))
        p["y"] += p["vel"]

        # Ações
        if acao == 1 and p["y"] > 40:
            p["vel"] = VELOCIDADE_PULO
            p["energia"] -= 0.025
        elif acao == -1 and p["y"] < CHAO - 40:
            p["vel"] = VELOCIDADE_DESCIDA
            p["energia"] -= 0.02
        else:
            p["energia"] -= 0.004

        # Campo simbiótico
        campo.evolve(p["energia"], random.uniform(-0.25, 0.25))
        energia_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
        p["energia"] = np.clip(p["energia"] + energia_local * 0.03, 0.0, 1.0)
        recompensa += energia_local * 2.0

        # Recompensa de estabilidade
        estabilidade = 1.0 - abs(p["y"] - ALTURA / 2) / (ALTURA / 2)
        recompensa += estabilidade * 0.5
        if p["y"] < ALTURA * 0.15 or p["y"] > ALTURA * 0.85:
            recompensa -= 0.6

        # Atualiza canos
        for cano in self.canos:
            cano["x"] -= VELOCIDADE_CANO_BASE

        ultimo_x = self.canos[-1]["x"] if self.canos else 0
        distancia_base = DISTANCIA_CANO_BASE + random.randint(50, 100)
        gap = random.randint(GAP_VERTICAL_MIN, GAP_VERTICAL_MAX)

        if not self.canos or ultimo_x < LARGURA - distancia_base:
            limite_inferior = 140 + gap // 2
            limite_superior = max(limite_inferior + 20, ALTURA - CHAO - gap // 2)
            nova_altura = random.randint(limite_inferior, limite_superior)
            novo_x = (self.canos[-1]["x"] + distancia_base) if self.canos else LARGURA + 100
            self.canos.append({"x": novo_x, "altura": nova_altura, "scored": False})

        self.canos = [c for c in self.canos if c["x"] > -120]

        # Pontuação
        for cano in self.canos:
            if not cano["scored"] and cano["x"] + 70 < p["x"]:
                cano["scored"] = True
                self.pontuacao += 1
                self.max_score = max(self.max_score, self.pontuacao)
                self._salvar_max_score()
                recompensa += 25
                self.imunidade_ativa = True
                self.imunidade_fim = agora + 1.0
                self.cano_seguro = cano
                p["vel"] *= 0.6

        # Remove imunidade após tempo
        if self.imunidade_ativa and agora > self.imunidade_fim:
            self.imunidade_ativa = False
            self.cano_seguro = None

        # Pega cano à frente
        canos_ativos = [c for c in self.canos if c["x"] > p["x"] - 50]
        cano_proximo = min(canos_ativos, key=lambda c: c["x"]) if canos_ativos else self.canos[0]

        # Colisão precisa
        cano_x, cano_h = cano_proximo["x"], cano_proximo["altura"]
        raio = 15
        passou_gap = (cano_h - 90 - raio) < p["y"] < (cano_h + 90 + raio)
        dentro_x = (cano_x - 30) < p["x"] < (cano_x + 100)
        colidiu = dentro_x and not passou_gap
        chao = p["y"] > (CHAO - 3)
        teto = p["y"] < 3

        if not self.imunidade_ativa and (colidiu or chao or teto):
            self.vivo = False
            self.cano_colidido = cano_proximo  # ← marca o cano atingido
            recompensa -= 300

        p["y"] = np.clip(p["y"], 0.0, CHAO - 1)
        terminado = not self.vivo
        return self._get_estado(acao), recompensa, terminado

    # ========================
    # Render
    # ========================
    def render(self, campo):
        now = time.time()
        if now - self.last_render < 1 / 60:
            return
        self.last_render = now

        surface = pygame.display.get_surface()
        if not surface:
            return

        energia_media = float(campo.field.mean().item())
        cor_ceu = (int(90 + energia_media * 50), int(130 + energia_media * 40), 255)
        surface.fill(cor_ceu)

        # === Desenha canos ===
        for cano in self.canos:
            cor_cano = (0, 200, 0)
            pygame.draw.rect(surface, cor_cano, (cano["x"], 0, 70, cano["altura"] - 90))
            pygame.draw.rect(surface, cor_cano, (cano["x"], cano["altura"] + 90, 70, ALTURA - cano["altura"]))

        # === Destaca o cano atingido ===
        if self.cano_colidido:
            hit_rect_top = pygame.Rect(self.cano_colidido["x"], 0, 70, self.cano_colidido["altura"] - 90)
            hit_rect_bottom = pygame.Rect(self.cano_colidido["x"], self.cano_colidido["altura"] + 90, 70,
                                          ALTURA - self.cano_colidido["altura"])
            overlay = pygame.Surface((70, ALTURA), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 120))  # vermelho translúcido
            surface.blit(overlay, (self.cano_colidido["x"], 0))

            pygame.draw.rect(surface, (255, 50, 50), hit_rect_top, 3)
            pygame.draw.rect(surface, (255, 50, 50), hit_rect_bottom, 3)

        # Chão
        pygame.draw.rect(surface, (222, 184, 135), (0, CHAO, LARGURA, 100))

        # Pássaro
        p = self.passaro
        energia_cor = int(255 * np.clip(p["energia"], 0, 1))
        pygame.draw.circle(surface, (255, energia_cor, 0),
                           (int(p["x"]), int(p["y"])), 15)

        # Texto
        fonte = pygame.font.SysFont("Arial", 20)
        texto = fonte.render(f"Pontuação: {self.pontuacao}", True, (255, 255, 255))
        recorde = fonte.render(f"🏆 Máxima: {self.max_score}", True, (255, 215, 0))
        surface.blit(texto, (10, 10))
        surface.blit(recorde, (10, 35))
        pygame.display.update()
