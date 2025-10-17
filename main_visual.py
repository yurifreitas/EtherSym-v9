# =====================================
# 🌈 EtherSym Visual Replay — Modo Show
# =====================================

import sys, re, json, pygame, torch, numpy as np
from pathlib import Path
from config import *
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from network import criar_modelo

# =====================================
# 📂 Utilidades
# =====================================
SAVE_PATH = Path("estado_treinamento.pth")
MAX_SCORE_PATH = Path("max_score.json")

def salvar_max_score(valor):
    try:
        with open(MAX_SCORE_PATH, "w") as f:
            json.dump({"max_score": valor}, f)
    except Exception as e:
        print(f"⚠️ Erro ao salvar max_score: {e}")

def carregar_max_score():
    if MAX_SCORE_PATH.exists():
        try:
            return json.load(open(MAX_SCORE_PATH)).get("max_score", -9999)
        except Exception:
            pass
    return -9999

# =====================================
# 🎬 Inicialização visual
# =====================================
pygame.init()
TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌠 EtherSym — Agente em Ação")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Consolas", 28, bold=True)
font_small = pygame.font.SysFont("Consolas", 16)

# paleta divertida
CORES = {
    "fundo": (10, 10, 20),
    "hud": (255, 255, 255),
    "brilho": (0, 180, 255),
    "particula": (255, 255, 120)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# =====================================
# 🧠 Modelo treinado
# =====================================
modelo, _, _, _ = criar_modelo(device)
try:
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("modelo", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {re.sub(r"^_orig_mod\.", "", k): v for k, v in state_dict.items()}
    modelo.load_state_dict(state_dict, strict=False)
    modelo.eval()
    print(f"✅ Modelo carregado: {SAVE_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {type(e).__name__}: {e}")
    sys.exit(1)

# =====================================
# 🌌 Ambiente e funções
# =====================================
campo = GravidadeAstrofisica()
env = AmbienteFlappy()
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

def escolher_acao_argmax(estado):
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q_vals = modelo(x)
    return ACTIONS[int(torch.argmax(q_vals, dim=1))]

# =====================================
# 💫 Sistema de partículas para diversão
# =====================================
particulas = []
def adicionar_particula(x, y):
    particulas.append([x, y, np.random.uniform(-2, 2), np.random.uniform(-4, -1), np.random.randint(3, 6)])

def atualizar_particulas():
    for p in particulas[:]:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= 0.15
        if p[4] <= 0:
            particulas.remove(p)
        else:
            pygame.draw.circle(TELA, CORES["particula"], (int(p[0]), int(p[1])), int(p[4]))

# =====================================
# 🕹️ Execução pura e divertida
# =====================================
estado = env.reset()
episodio, total_recompensa, melhor = 0, 0.0, carregar_max_score()
FPS_REAL = 60
brilho = 0

print("🎮 Iniciando modo visual... Divirta-se!")

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    # === Ação ===
    acao = escolher_acao_argmax(estado)
    novo_estado, recompensa, terminado = env.step(acao, campo)
    total_recompensa += recompensa
    estado = novo_estado

    # === Renderização ===
    TELA.fill(CORES["fundo"])
    env.render(campo)

    # partículas simbólicas de energia
    if np.random.random() < 0.4:
        adicionar_particula(200 + np.random.randint(-60, 60), 300 + np.random.randint(-50, 50))
    atualizar_particulas()

    # brilho pulsante
    brilho = (brilho + 4) % 360
    cor_brilho = (
        int(128 + 127 * np.sin(np.radians(brilho))),
        int(128 + 127 * np.sin(np.radians(brilho + 120))),
        int(128 + 127 * np.sin(np.radians(brilho + 240))),
    )

    # HUD principal
    texto1 = f"Ep {episodio:04d} | Reward={total_recompensa:7.1f} | Melhor={melhor:7.1f}"
    pygame.draw.rect(TELA, (0, 0, 0, 100), (0, 0, LARGURA, 40))
    TELA.blit(font_small.render(texto1, True, CORES["hud"]), (10, 10))

    pygame.draw.circle(TELA, cor_brilho, (LARGURA - 80, 30), 12)
    pygame.display.flip()
    clock.tick(FPS_REAL)

    # === Episódio concluído ===
    if terminado:
        melhor = max(melhor, total_recompensa)
        salvar_max_score(float(melhor))
        print(f"🏁 Ep {episodio:04d} — Reward={total_recompensa:.1f} | Melhor={melhor:.1f}")
        estado = env.reset()
        total_recompensa = 0.0
        episodio += 1
