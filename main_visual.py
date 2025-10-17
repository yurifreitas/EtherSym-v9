# =====================================
# 🌈 EtherSym Visual Replay — Pontuação por Cano (HUD Funcional)
# =====================================

import sys, re, json, pygame, torch, numpy as np
from pathlib import Path
from config import *
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from network import criar_modelo

# =====================================
# 📂 Persistência do Max Score
# =====================================
SAVE_PATH = Path("estado_treinamento.pth")
MAX_SCORE_PATH = Path("max_score.json")

def salvar_max_score(path, valor):
    try:
        with open(path, "w") as f:
            json.dump({"max_score": valor}, f)
    except Exception as e:
        print(f"⚠️ Erro ao salvar max_score: {e}")

def carregar_max_score(path):
    if Path(path).exists():
        try:
            return json.load(open(path)).get("max_score", 0)
        except Exception:
            pass
    return 0

# =====================================
# 🎬 Inicialização
# =====================================
pygame.init()
TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌠 EtherSym — Pontuação por Cano")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Consolas", 36, bold=True)
font_small = pygame.font.SysFont("Consolas", 20)

CORES = {
    "fundo": (10, 10, 20),
    "hud": (255, 255, 255),
    "max": (255, 255, 120),
    "atual": (80, 200, 255),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# =====================================
# 🧠 Modelo
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
# 🌌 Ambiente
# =====================================
campo = GravidadeAstrofisica()
env = AmbienteFlappy()
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

@torch.no_grad()
def escolher_acao_argmax(estado):
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q_vals = modelo(x)
    return ACTIONS[int(torch.argmax(q_vals, dim=1))]

# =====================================
# 🕹️ Loop Principal
# =====================================
estado = env.reset()
FPS_REAL = 60

# usa pontuação do ambiente diretamente
pontuacao = env.pontuacao
melhor = carregar_max_score(env.max_score_path)

print(f"🎮 Modo Show iniciado — Max Score atual: {melhor}")

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    # === Passo da IA ===
    acao = escolher_acao_argmax(estado)
    novo_estado, recompensa, terminado = env.step(acao, campo)
    estado = novo_estado

    # === Atualiza pontuação e recorde ===
    pontuacao = env.pontuacao
    if pontuacao > melhor:
        melhor = pontuacao
        salvar_max_score(env.max_score_path, melhor)
        print(f"🌟 Novo recorde! {melhor}")

    # === Renderização ===
    TELA.fill(CORES["fundo"])
    env.render(campo)

    # HUD fixo
    pygame.draw.rect(TELA, (0, 0, 0, 100), (0, 0, LARGURA, 60))
    texto_max = font_small.render(f"🌟 MAX: {int(melhor)}", True, CORES["max"])
    texto_atual = font_big.render(f"🏆 {int(pontuacao)}", True, CORES["atual"])
    TELA.blit(texto_max, (10, 5))
    TELA.blit(texto_atual, (10, 25))

    pygame.display.flip()
    clock.tick(FPS_REAL)

    # === Episódio concluído ===
    if terminado:
        print(f"🏁 Fim | Pontuação={pontuacao} | Recorde={melhor}")
        estado = env.reset()
        pontuacao = 0
