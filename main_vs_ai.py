# ==========================================
# 🌌 EtherSym v9.9 — Duelo Evolutivo Real (Humano vs IA)
# ==========================================
# Ambos no mesmo campo — mostra pontuação, diferença e recordes.
# Corrigido para compatibilidade com self.passaro.y
# ==========================================

import pygame, torch, numpy as np, re, sys
from config import *
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from network import criar_modelo

# ==========================================
# ⚙️ Inicialização
# ==========================================
pygame.init()
TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌌 EtherSym — Duelo Evolutivo: Humano 🧍 vs IA 🤖")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# ==========================================
# 🧠 Carregando o modelo IA simbiótica
# ==========================================
modelo, _, _, _ = criar_modelo(device)
try:
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    if "modelo" in ckpt:
        state_dict = ckpt["modelo"]
    else:
        state_dict = ckpt
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {re.sub(r"^_orig_mod\.", "", k): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict, strict=False)
    modelo.eval()
    torch.set_grad_enabled(False)
    print(f"✅ Modelo carregado de {SAVE_PATH}")
except Exception as e:
    print(f"⚠️ Falha ao carregar IA treinada: {e}")
    modelo.eval()

# ==========================================
# 🌌 Campo físico compartilhado
# ==========================================
campo = GravidadeAstrofisica()
env = AmbienteFlappy()

ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

def escolher_acao_ia(estado):
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q = modelo(x)
    return ACTIONS[int(torch.argmax(q, dim=1))]

# ==========================================
# 🧩 Adaptações locais seguras
# ==========================================
def get_passaro_y(env):
    """Retorna posição Y do pássaro (segura para qualquer estrutura)."""
    try:
        if hasattr(env, "passaro") and hasattr(env.passaro, "y"):
            return env.passaro.y
        elif hasattr(env, "y"):
            return env.y
        else:
            return ALTURA // 2
    except Exception:
        return ALTURA // 2

def step_duplo(env, acao_humano, acao_ia, campo):
    """Executa passo simultâneo (humano e IA) sem interferência."""
    estado_atual = env.get_state() if hasattr(env, "get_state") else None
    if estado_atual is not None and hasattr(env, "set_state"):
        env.set_state(estado_atual)
    novo_estado_humano, r1, t1 = env.step(acao_humano, campo)

    if estado_atual is not None and hasattr(env, "set_state"):
        env.set_state(estado_atual)
    novo_estado_ia, r2, t2 = env.step(acao_ia, campo)
    return (novo_estado_humano, novo_estado_ia), (r1, r2), (t1, t2)

def render_duelo(env, campo, y_humano, y_ia):
    """Renderiza os dois jogadores no mesmo campo."""
    env.render(campo)
    surf = pygame.display.get_surface()
    pygame.draw.circle(surf, (0, 180, 255), (int(env.passaro_x), int(y_humano)), 10)
    pygame.draw.circle(surf, (255, 200, 0), (int(env.passaro_x) + 10, int(y_ia)), 10)

# ==========================================
# 🎮 Inicialização dos agentes
# ==========================================
estado_base = env.reset()
estado_ia = np.copy(estado_base)
estado_humano = np.copy(estado_base)

player_alive = ai_alive = True
score_player = score_ai = 0.0
melhor_player = melhor_ai = -9999.0
y_humano = get_passaro_y(env)
y_ia = get_passaro_y(env)

FPS_REAL = 60
print("🎮 Duelo simbiótico iniciado — mesmo universo físico: 🧍 Humano vs 🤖 IA")

# ==========================================
# 🚀 Loop principal
# ==========================================
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    # === Entrada humana ===
    keys = pygame.key.get_pressed()
    acao_humano = 0
    if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
        acao_humano = 1
    elif keys[pygame.K_DOWN]:
        acao_humano = -1

    # === Ação da IA ===
    acao_ia = escolher_acao_ia(estado_ia)

    # === Passo compartilhado ===
    (novo_estado_humano, novo_estado_ia), (r1, r2), (t1, t2) = step_duplo(env, acao_humano, acao_ia, campo)

    score_player += r1
    score_ai += r2
    estado_humano, estado_ia = novo_estado_humano, novo_estado_ia
    y_humano = get_passaro_y(env)
    y_ia = get_passaro_y(env)

    # --- Renderiza ---
    render_duelo(env, campo, y_humano, y_ia)
    txt1 = font.render(f"🧍 Humano: {int(score_player)}", True, (0, 200, 255))
    txt2 = font.render(f"🤖 IA: {int(score_ai)}", True, (255, 200, 0))
    diff = score_ai - score_player
    if diff > 0:
        status = f"🤖 IA lidera (+{abs(diff):.0f})"
    elif diff < 0:
        status = f"🧍 Humano lidera (+{abs(diff):.0f})"
    else:
        status = "🏁 Empate técnico"
    txt3 = font.render(status, True, (255, 255, 255))

    TELA.blit(txt1, (10, 10))
    TELA.blit(txt2, (10, 30))
    TELA.blit(txt3, (10, 50))
    pygame.display.flip()
    clock.tick(FPS_REAL)

    # --- Reinício ---
    if t1 or t2:
        melhor_player = max(melhor_player, score_player)
        melhor_ai = max(melhor_ai, score_ai)

        print(f"🏁 Fim — 🧍={score_player:.1f} | 🤖={score_ai:.1f}")
        print(f"📈 Recordes — Humano={melhor_player:.1f} | IA={melhor_ai:.1f}")

        estado_base = env.reset()
        estado_ia = np.copy(estado_base)
        estado_humano = np.copy(estado_base)
        score_player = score_ai = 0.0
        player_alive = ai_alive = True
        y_humano = y_ia = get_passaro_y(env)
