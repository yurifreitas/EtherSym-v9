import sys, re, json, pygame, torch, numpy as np
from pathlib import Path
from config import *
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from network import criar_modelo

# =====================================
# 📂 Caminhos e utilidades
# =====================================
MAX_SCORE_PATH = Path("max_score.json")

def salvar_max_score(valor):
    """Salva o melhor score atual de forma segura."""
    try:
        with open(MAX_SCORE_PATH, "w") as f:
            json.dump({"max_score": valor}, f)
    except Exception as e:
        print(f"⚠️ Erro ao salvar max_score: {e}")

# =====================================
# 🎬 Inicialização
# =====================================
pygame.init()
TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌌 EtherSym — Agente Treinado")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# =====================================
# 🧠 Modelo treinado
# =====================================
modelo, alvo, opt, _ = criar_modelo(device)

try:
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    if "modelo" in ckpt:
        state_dict = ckpt["modelo"]
    else:
        state_dict = ckpt  # fallback: se for state_dict direto

    # Remove prefixo "_orig_mod." (compatibilidade com torch.compile)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {re.sub(r"^_orig_mod\.", "", k): v for k, v in state_dict.items()}

    missing, unexpected = modelo.load_state_dict(state_dict, strict=False)

    modelo.eval()
    torch.set_grad_enabled(False)

    print(f"✅ Modelo carregado: {SAVE_PATH}")
    print(f"🔸 Pesos carregados: {len(state_dict)}")
    print(f"🔹 Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print(f"ε final: {ckpt.get('epsilon', 'N/A')}")
    print(f"Média recompensa (treino): {ckpt.get('media_recompensa', 'N/A')}")

except Exception as e:
    print(f"❌ Erro ao carregar modelo: {type(e).__name__}: {e}")
    sys.exit(1)

# =====================================
# 🌌 Ambiente
# =====================================
campo = GravidadeAstrofisica()
env = AmbienteFlappy()
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

def escolher_acao_argmax(estado):
    """Retorna a ação com maior valor Q."""
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q_vals = modelo(x)
    return ACTIONS[int(torch.argmax(q_vals, dim=1))]

# =====================================
# 🕹️ Execução pura (modo visual)
# =====================================
estado = env.reset()
episodio, total_recompensa, melhor = 0, 0.0, -9999.0
FPS_REAL = 60

print("🎬 Iniciando modo visual (replay do agente)...")

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    acao = escolher_acao_argmax(estado)
    novo_estado, recompensa, terminado = env.step(acao, campo)
    total_recompensa += recompensa
    estado = novo_estado

    # --- renderização ---
    env.render(campo)
    texto = f"Ep {episodio:04d} | Reward={total_recompensa:7.1f} | Melhor={melhor:7.1f}"
    TELA.blit(font.render(texto, True, (255, 255, 255)), (10, 10))
    pygame.display.flip()
    clock.tick(FPS_REAL)

    if terminado:
        melhor = max(melhor, total_recompensa)
        salvar_max_score(float(melhor))
        print(f"🏁 Ep {episodio:04d} — Reward={total_recompensa:.1f} | Melhor={melhor:.1f}")
        estado = env.reset()
        total_recompensa = 0.0
        episodio += 1
