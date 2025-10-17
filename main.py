# main_watch.py — ver o agente jogar usando o modelo treinado (sem treino)

import sys, pygame, torch, numpy as np
from config import *                      # reaproveita LARGURA, ALTURA, FPS, etc.
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from memory import carregar_estado
from network import criar_modelo

# ---------- janela ----------
pygame.init()
pygame.font.init()
TELA = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("🌌 EtherSym — Agente Treinado (Somente Replay)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

# ---------- aceleração (segura para inferência) ----------
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- carrega modelo treinado ----------
modelo, _, opt, _ = criar_modelo(device)   # cria estrutura
_, EPSILON, media_recompensa = carregar_estado(modelo, opt)  # carrega pesos salvos
modelo.eval()                               # IMPORTANTÍSSIMO: desativa dropout etc.
torch.set_grad_enabled(False)               # garante que nada treina

print("✅ Modelo carregado!")
print(f"📊 Média de recompensa (no treino): {float(media_recompensa or 0):.2f}")

# ---------- ambiente ----------
campo = GravidadeAstrofisica()
env   = AmbienteFlappy()

ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

def escolher_acao_argmax(estado):
    """Política determinística (sem exploração) a partir do modelo treinado."""
    x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
    q = modelo(x)                        # [1, 3]
    a_idx = int(torch.argmax(q, dim=1))  # 0..2
    return ACTIONS[a_idx]                # {-1,0,1}

# ---------- loop de visualização (sem treino) ----------
estado = env.reset()
total_recompensa = 0.0
melhor = -1e9
episodio = 0
VISUAL_FPS = max(1, FPS or 60)           # usa FPS do config, fallback 60

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # ação da política treinada (SEM epsilon, SEM softmax)
    acao = escolher_acao_argmax(estado)
    novo_estado, recompensa, terminado = env.step(acao, campo)
    total_recompensa += float(recompensa)
    estado = novo_estado

    # render / overlay
    env.render(campo)
    overlay = f"Ep {episodio:04d} | Reward={total_recompensa:7.1f} | Média(treino)={float(media_recompensa or 0):7.1f}"
    txt = font.render(overlay, True, (255,255,255))
    TELA.blit(txt, (10, 10))

    pygame.display.flip()
    clock.tick(VISUAL_FPS)               # controla a velocidade (tempo normal)

    # fim do episódio (reinicia só para continuar assistindo)
    if terminado:
        melhor = max(melhor, total_recompensa)
        print(f"🏁 Ep {episodio:04d} — Reward={total_recompensa:.1f} | Melhor={melhor:.1f}")
        estado = env.reset()
        total_recompensa = 0.0
        episodio += 1

pygame.quit()
sys.exit(0)
