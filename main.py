# ==========================================
# 🌌 EtherSym — Modo Turbo Treinado + Replay
# ==========================================

import os, sys, pygame, torch, numpy as np
from config import *  # LARGURA, ALTURA, FPS, SAVE_PATH
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from network import criar_modelo

# ==========================================
# ⚙️ Parâmetros de Modo Turbo
# ==========================================
FAST_MODE = True            # True = mais rápido (menos render)
ULTRA_TURBO = False         # True = sem render nenhum
STEPS_PER_RENDER = 8        # passos por frame de renderização
ACTION_REPEAT = 4           # repetir mesma ação X vezes
RENDER_INTERVAL = 10        # renderiza a cada X steps
FPS = 120                   # limite visual máximo
SHOW_OVERLAY = True         # mostrar texto na tela
TOGGLE_KEYS = True          # ativa atalhos (T e R)

# ==========================================
# 🧠 Configuração do Modelo
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, SAVE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True

# ==========================================
# 🧩 Carrega Modelo
# ==========================================
modelo, _, _, _ = criar_modelo(device)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "modelo" in checkpoint:
        modelo.load_state_dict(checkpoint["modelo"], strict=False)
        media_recompensa = checkpoint.get("media_recompensa", 0.0)
    else:
        modelo.load_state_dict(checkpoint, strict=False)
        media_recompensa = 0.0
    print(f"✅ Modelo carregado: {MODEL_PATH}")
    print(f"📊 Média de recompensa (treino): {media_recompensa:.2f}")
except Exception as e:
    print(f"⚠️ Erro ao carregar modelo: {type(e).__name__}: {e}")
    sys.exit(1)

modelo.eval()
torch.set_grad_enabled(False)

# ==========================================
# 🎮 Ambiente Pygame
# ==========================================
pygame.init()
pygame.font.init()

TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌌 EtherSym — Modo Turbo / Replay")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

campo = GravidadeAstrofisica()
env = AmbienteFlappy()
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

# ==========================================
# 🔮 Funções auxiliares
# ==========================================
def escolher_acao_argmax(estado):
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q = modelo(x)
    return ACTIONS[int(torch.argmax(q, dim=1))]

def render_overlay(episodio, total, media, turbo, melhor):
    if not SHOW_OVERLAY or ULTRA_TURBO:
        return
    turbo_txt = "⚡" if turbo else ""
    overlay = (
        f"{turbo_txt} Ep {episodio:04d} | Reward={total:7.1f} "
        f"| Média={media:7.1f} | Melhor={melhor:7.1f}"
    )
    txt = font.render(overlay, True, (255, 255, 255))
    TELA.blit(txt, (10, 10))

# ==========================================
# 🔁 Loop principal
# ==========================================
estado = env.reset()
total_recompensa = 0.0
melhor = -1e9
episodio = 0
step = 0

print("🎬 Iniciando replay / treinamento turbo...")

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif TOGGLE_KEYS and e.type == pygame.KEYDOWN:
            if e.key == pygame.K_t:  # alternar turbo
                FAST_MODE = not FAST_MODE
                ULTRA_TURBO = False
                print(f"⚙️ FAST_MODE = {FAST_MODE}")
            elif e.key == pygame.K_r:  # reiniciar episódio
                estado = env.reset()
                total_recompensa = 0.0
                print(f"🔄 Episódio reiniciado ({episodio})")

    # --- ação determinística ---
    acao = escolher_acao_argmax(estado)
    for _ in range(ACTION_REPEAT):
        novo_estado, recompensa, terminado = env.step(acao, campo)
        total_recompensa += float(recompensa)
        estado = novo_estado
        step += 1
        if terminado:
            break

    # --- renderização ---
    if not ULTRA_TURBO:
        if not FAST_MODE or (step % RENDER_INTERVAL == 0):
            env.render(campo)
            render_overlay(episodio, total_recompensa, media_recompensa, FAST_MODE, melhor)
            pygame.display.flip()

    if not FAST_MODE:
        clock.tick(FPS)

    # --- fim do episódio ---
    if terminado:
        melhor = max(melhor, total_recompensa)
        print(f"🏁 Ep {episodio:04d} — Reward={total_recompensa:.1f} | Melhor={melhor:.1f}")
        episodio += 1
        total_recompensa = 0.0
        estado = env.reset()
        step = 0

pygame.quit()
sys.exit(0)
