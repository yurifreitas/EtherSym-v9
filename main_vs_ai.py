# ==========================================
# 🌌 EtherSym v9 — Turbo Dueling (Humano vs IA)
# ==========================================

import os, sys, random, numpy as np, pygame, torch, re
from config import (
    LARGURA, ALTURA, FPS, FAST_MODE, RENDER_INTERVAL,
    STEPS_PER_RENDER, ACTION_REPEAT, SAVE_PATH
)
from field import GravidadeAstrofisica
from physics.duel_env import DuelEnv              # <- ambiente de duelo
from network import criar_modelo

# ============== Turbo CUDA / determinismo leve ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
except Exception:
    pass

def reseed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
reseed(42)

# =================== Janela (HEADLESS opcional) ===================
if os.environ.get("HEADLESS", "0") == "1":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init(); pygame.font.init()
DISPLAY_FLAGS = pygame.DOUBLEBUF | pygame.HWSURFACE
try:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS, vsync=0)
except Exception:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS)
pygame.display.set_caption("🌌 EtherSym v9 — Duelo Turbo (Humano 🧍 vs 🤖 IA)")
clock = pygame.time.Clock()
pygame.event.set_allowed([pygame.QUIT])  # evita fila lotada

font = pygame.font.SysFont("Consolas", 18)

# =================== Modelo ===================
modelo, _, _, _ = criar_modelo(device)
try:
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["modelo"] if "modelo" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {re.sub(r"^_orig_mod\.", "", k): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict, strict=False)
    print(f"✅ Modelo carregado de {SAVE_PATH}")
except Exception as e:
    print(f"⚠️ Falha ao carregar IA treinada: {e}")
modelo.eval(); torch.set_grad_enabled(False)

# Compilação adaptativa (se CUDA)
try:
    if device.type == "cuda":
        modelo = torch.compile(modelo, mode="max-autotune")
except Exception:
    pass

ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
TEMPERATURA = 0.85  # mesma vibe do treino

@torch.no_grad()
def escolher_acao_ia(estado):
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    logits = modelo(x)
    probs = torch.softmax(logits / TEMPERATURA, dim=1).float()
    p = probs.clamp(1e-6, 1.0).cpu().numpy().ravel()
    p = p / p.sum()
    return int(np.random.choice(ACTIONS, p=p))

# =================== Ambiente ===================
campo = GravidadeAstrofisica()
env   = DuelEnv()       # precisa do duel_env com advance_world corrigido
env.reset_duel()

x_ref = getattr(env, "passaro_x", 80)

# =================== Estado/HUD ===================
score_h = score_ai = 0.0
best_h = best_ai = -1e9
running = True
print("🎮 Duelo simbiótico (Turbo) iniciado — mesmo universo físico: 🧍 Humano vs 🤖 IA")

def render_hud():
    diff = score_ai - score_h
    if diff > 0:   status = f"🤖 IA lidera (+{abs(diff):.0f})"
    elif diff < 0: status = f"🧍 Humano lidera (+{abs(diff):.0f})"
    else:          status = "🏁 Empate técnico"
    TELA.blit(font.render(f"🧍 Humano: {int(score_h)}", True, (0, 200, 255)), (10, 10))
    TELA.blit(font.render(f"🤖 IA: {int(score_ai)}", True, (255, 200, 0)), (10, 30))
    TELA.blit(font.render(status, True, (255, 255, 255)), (10, 50))

# =================== Loop principal ===================
while running:
    # eventos mínimos
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # —— passos turbo entre renders ——
    steps = max(1, STEPS_PER_RENDER)
    for _ in range(steps):
        keys = pygame.key.get_pressed()
        a_h = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) else (-1 if keys[pygame.K_DOWN] else 0)
        estado_ia = env._encode_obs(env.ai)   # shape (6,)
        a_ai = escolher_acao_ia(estado_ia)

        (s_h, s_ai), (r_h, r_ai), (d_h, d_ai) = env.step_duel(a_h, a_ai, campo)
        score_h  += float(r_h)
        score_ai += float(r_ai)

        if d_h or d_ai:
            best_h = max(best_h, score_h)
            best_ai = max(best_ai, score_ai)
            print(f"🏁 Fim — 🧍={score_h:.1f} | 🤖={score_ai:.1f} | 📈 Recordes — H={best_h:.1f} IA={best_ai:.1f}")
            env.reset_duel()
            score_h = score_ai = 0.0
            break

    # —— renderização controlada —— 
    if not FAST_MODE:
        env.render(campo)
        # desenha os dois pássaros sobre o render base
        surf = pygame.display.get_surface()
        pygame.draw.circle(surf, (0, 180, 255), (int(x_ref), int(env.human.y)), 10)
        pygame.draw.circle(surf, (255, 200, 0), (int(x_ref) + 10, int(env.ai.y)), 10)
        render_hud()
        pygame.display.flip()
        clock.tick(FPS or 0)
    else:
        # mesmo em FAST_MODE, renderiza de tempos em tempos
        if RENDER_INTERVAL > 0 and (pygame.time.get_ticks() % RENDER_INTERVAL == 0):
            env.render(campo)
            surf = pygame.display.get_surface()
            pygame.draw.circle(surf, (0, 180, 255), (int(x_ref), int(env.human.y)), 10)
            pygame.draw.circle(surf, (255, 200, 0), (int(x_ref) + 10, int(env.ai.y)), 10)
            render_hud()
            pygame.display.flip()

# —— encerramento limpo ——
try:
    TELA.fill((0,0,0)); pygame.display.flip()
except Exception:
    pass
pygame.time.delay(60)
pygame.quit()
sys.exit(0)
