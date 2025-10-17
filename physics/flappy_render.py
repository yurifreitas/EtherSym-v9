import pygame
import numpy as np
import time
from config import *


def renderizar_ambiente(env, campo):
    surface = pygame.display.get_surface()
    if not surface:
        return

    # === Fundo fractal hipnótico ===
    energia_media = float(campo.field.mean().item()) if hasattr(campo, "field") else 0.5
    if hasattr(env, "bg") and env.bg is not None:
        env.bg.render(surface, energia_media, time.time())
    else:
        # fallback (nunca deve acontecer, mas evita tela preta)
        cor_ceu = (int(90 + energia_media * 50), int(130 + energia_media * 40), 255)
        surface.fill(cor_ceu)

    # === Canos ===
    for cano in env.canos:
        pygame.draw.rect(
            surface,
            (0, 200, 0),
            (cano["x"], 0, 70, cano["altura"] - 90),
        )
        pygame.draw.rect(
            surface,
            (0, 200, 0),
            (cano["x"], cano["altura"] + 90, 70, ALTURA - cano["altura"]),
        )

    # === Destaque do cano colidido (debug visual) ===
    if getattr(env, "cano_colidido", None):
        overlay = pygame.Surface((70, ALTURA), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 100))
        surface.blit(overlay, (env.cano_colidido["x"], 0))
        pygame.draw.rect(
            surface,
            (255, 60, 60),
            (env.cano_colidido["x"], 0, 70, ALTURA),
            3,
        )

    # === Chão ===
    pygame.draw.rect(surface, (222, 184, 135), (0, CHAO, LARGURA, 100))

   # === 🕊️ Pássaro simbiótico estilizado ===
    p = env.passaro
    x, y = int(p["x"]), int(p["y"])
    energia = np.clip(p["energia"], 0, 1)

    # Cores baseadas na energia simbiótica
    cor_base = (255, int(180 + 75 * energia), int(80 + 120 * (1 - energia)))
    cor_asa  = (255, int(200 * energia), 40)
    cor_olho = (255, 255, 255)
    cor_pupila = (0, 0, 0)
    cor_bico = (255, 180, 0)

    # Corpo ovalado
    pygame.draw.ellipse(surface, cor_base, (x - 18, y - 12, 36, 24))

    # Asa dinâmica (bate suavemente com o tempo)
    t = time.time() * 6.0
    batida = np.sin(t) * 6
    pygame.draw.polygon(
        surface,
        cor_asa,
        [
            (x - 6, y - 4),
            (x - 26, y - 10 - batida),
            (x - 10, y + 4 + batida),
        ],
    )

    # Cabeça
    pygame.draw.circle(surface, cor_base, (x + 14, y - 4), 10)

    # Olho
    pygame.draw.circle(surface, cor_olho, (x + 17, y - 6), 4)
    pygame.draw.circle(surface, cor_pupila, (x + 17, y - 6), 2)

    # Bico triangular (pequeno, voltado pra frente)
    pygame.draw.polygon(
        surface,
        cor_bico,
        [
            (x + 24, y - 3),
            (x + 32, y),
            (x + 24, y + 3),
        ],
    )

    # Halo energético pulsante (quando energia > 0.9)
    if energia > 0.9:
        raio = int(25 + 5 * np.sin(time.time() * 5))
        cor_halo = (255, int(200 + 55 * np.sin(time.time() * 2)), 80)
        pygame.draw.circle(surface, cor_halo, (x, y), raio, width=2)


    # === HUD ===
    fonte = pygame.font.SysFont("Arial", 20)
    texto = fonte.render(f"Pontuação: {env.pontuacao}", True, (255, 255, 255))
    recorde = fonte.render(f"🏆 Máxima: {env.max_score}", True, (255, 215, 0))
    surface.blit(texto, (10, 10))
    surface.blit(recorde, (10, 35))

    # ❌ NÃO ATUALIZE A TELA AQUI
    # pygame.display.update()  ← REMOVIDO para evitar flicker
