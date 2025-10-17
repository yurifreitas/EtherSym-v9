# ==========================================
# 🌌 EtherSym v9.9 — Modo Solo (Somente Humano 🧍)
# ==========================================

import os, sys, random, numpy as np, pygame, torch, re, json
from config import LARGURA, ALTURA, SAVE_PATH, FAST_MODE
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy

# ==========================================
# ⚙️ Inicialização
# ==========================================
pygame.init()
pygame.font.init()
flags = pygame.DOUBLEBUF | pygame.HWSURFACE
TELA = pygame.display.set_mode((LARGURA, ALTURA), flags)
pygame.display.set_caption("🌌 EtherSym — Modo Solo 🧍 (Somente Humano)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 20)
pygame.event.set_allowed([pygame.QUIT])

# ==========================================
# 🎯 Funções utilitárias
# ==========================================
MAX_SCORE_PATH = "max_score.json"

def salvar_max_score(valor):
    try:
        with open(MAX_SCORE_PATH, "w") as f:
            json.dump({"max_score": valor}, f)
    except Exception as e:
        print(f"⚠️ Erro ao salvar max_score: {e}")

def carregar_max_score():
    if os.path.exists(MAX_SCORE_PATH):
        try:
            return json.load(open(MAX_SCORE_PATH)).get("max_score", 0)
        except Exception:
            pass
    return 0

# ==========================================
# 🌌 Ambiente
# ==========================================
campo = GravidadeAstrofisica()
env = AmbienteFlappy()
env.reset()
melhor = carregar_max_score()

# ==========================================
# 🧍 Estado inicial
# ==========================================
pontuacao = 0
ultimo_cano_x = None
FPS = 45
running = True
print(f"🎮 Modo Solo iniciado — Max Score atual: {melhor}")

# ==========================================
# 🧾 HUD
# ==========================================
def render_hud():
    surf = pygame.display.get_surface()
    texto_max = font.render(f"🌟 RECORD: {int(melhor)}", True, (255, 255, 100))
    texto_atual = font.render(f"🏆 SCORE: {int(pontuacao)}", True, (80, 200, 255))
    surf.blit(texto_max, (10, 10))
    surf.blit(texto_atual, (10, 35))

# ==========================================
# 🚀 Loop principal
# ==========================================
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # === Entrada do jogador ===
    keys = pygame.key.get_pressed()
    acao = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) else 0

    # === Atualiza ambiente ===
    estado, recompensa, terminou = env.step(acao, campo)
    env.render(campo)

    # === Detecta novo cano atravessado ===
    if hasattr(env, "canos") and len(env.canos) > 0:
        cano = env.canos[0]
        x_cano = cano["x"]
        x_passaro = env.passaro["x"]

        if ultimo_cano_x is None or (x_cano < x_passaro and x_cano != ultimo_cano_x):
            pontuacao += 1
            ultimo_cano_x = x_cano
            print(f"🎯 +1 ponto! total={pontuacao}")

            if pontuacao > melhor:
                melhor = pontuacao
                salvar_max_score(melhor)
                print(f"🌟 Novo recorde: {melhor}")

    # === Renderização ===
    render_hud()
    pygame.display.flip()
    clock.tick(FPS)

    # === Fim de jogo ===
    if terminou:
        print(f"💀 Fim de jogo | Pontuação={pontuacao} | Recorde={melhor}")
        pygame.time.delay(1000)
        pontuacao = 0
        ultimo_cano_x = None
        env.reset()

# ==========================================
# Encerramento
# ==========================================
TELA.fill((0, 0, 0))
pygame.display.flip()
pygame.quit()
sys.exit(0)
