# ==========================================
# 🌌 EtherSym v9 — Gravidade Astrofísica e Neurogênese Simbiótica
# Versão estável (com borda, sem flicker, encerramento limpo)
# ==========================================

import pygame, sys, random, torch, numpy as np
from collections import deque

from config import *
from utils import reseed, warmup
from field import GravidadeAstrofisica
from network import criar_modelo
from physics.flappy_env import AmbienteFlappy
from memory import salvar_estado, carregar_estado

# ========================
# Inicialização da Janela
# ========================
pygame.init()
pygame.font.init()

# Flags seguras e suaves (sem NOFRAME)
DISPLAY_FLAGS = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.SCALED

# VSync melhora a estabilidade (quando suportado)
try:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS, vsync=1)
except TypeError:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS)

pygame.display.set_caption("🌌 EtherSym v9 — Gravidade Astrofísica e Neurogênese Simbiótica")
clock = pygame.time.Clock()

# ========================
# Modelo e Memória
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo, alvo, otimizador, loss_fn = criar_modelo(device)
memoria, EPSILON, media_recompensa = carregar_estado(modelo, otimizador)

campo = GravidadeAstrofisica()
env   = AmbienteFlappy()

# ========================
# Hiperparâmetros
# ========================
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
TEMPERATURA_BASE = 0.95
TEMPERATURA_MIN  = 0.60
TARGET_TAU       = 0.02
TARGET_SYNC_HARD = 2000
AUTOSAVE_EVERY   = 2000
MIN_REPLAY       = max(BATCH * 5, 1000)
global_step      = 0
hard_sync_step   = 0

# ========================
# Escolha de ação (ε-greedy + softmax)
# ========================
def escolher_acao(estado):
    global EPSILON
    if random.random() < EPSILON:
        return int(np.random.choice(ACTIONS))
    with torch.no_grad():
        x = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
        logits = modelo(x)
        temp = max(TEMPERATURA_MIN, TEMPERATURA_BASE * (0.8 + 0.2 * (EPSILON / max(EPSILON_MIN, EPSILON))))
        probs = torch.softmax(logits / temp, dim=1).float().clamp(1e-6, 1.0)
        probs = (probs / probs.sum(dim=1, keepdim=True)).cpu().numpy().ravel()
        return int(np.random.choice(ACTIONS, p=probs))

# ========================
# Atualização suave (Polyak)
# ========================
@torch.no_grad()
def soft_update(alvo, online, tau=TARGET_TAU):
    for p_t, p in zip(alvo.parameters(), online.parameters()):
        p_t.data.mul_(1.0 - tau).add_(tau * p.data)

# ========================
# Loop principal
# ========================
estado = warmup(env, campo)
total_recompensa = 0.0
epoch = 0
recompensas = deque(maxlen=120)

pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])

while True:
    # === Eventos ===
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            print("🛑 Encerrando simulação EtherSym...")
            # limpa a tela antes de sair (evita embaçado)
            TELA.fill((0, 0, 0))
            pygame.display.flip()
            salvar_estado(modelo, otimizador, memoria, EPSILON, media_recompensa)
            pygame.time.delay(150)
            pygame.quit()
            sys.exit()

    # === Passo simbiótico ===
    acao = escolher_acao(estado)
    novo_estado, recompensa, terminado = env.step(acao, campo)

    recompensa = float(np.clip(recompensa, -300.0, 50.0))
    memoria.append((estado, acao, recompensa, novo_estado, terminado))
    total_recompensa += recompensa
    estado = novo_estado
    global_step += 1
    hard_sync_step += 1

    # === Treinamento ===
    if len(memoria) >= MIN_REPLAY:
        lote = random.sample(memoria, BATCH)
        estados, acoes, recompensas_lote, novos_estados, finais = zip(*lote)

        estados_t       = torch.tensor(np.asarray(estados, dtype=np.float32), device=device)
        novos_estados_t = torch.tensor(np.asarray(novos_estados, dtype=np.float32), device=device)
        acoes_t         = torch.tensor(np.asarray(acoes, dtype=np.int64), device=device).unsqueeze(1)
        recompensas_t   = torch.tensor(np.asarray(recompensas_lote, dtype=np.float32), device=device)
        finais_t        = torch.tensor(np.asarray(finais, dtype=np.float32), device=device)

        idx_t = acoes_t + 1

        with torch.no_grad():
            next_online_q = modelo(novos_estados_t)
            next_actions  = torch.argmax(next_online_q, dim=1)
            next_target_q = alvo(novos_estados_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            alvo_q        = recompensas_t + GAMMA * next_target_q * (1.0 - finais_t)

        q_vals = modelo(estados_t).gather(1, idx_t).squeeze(1)
        perda = loss_fn(q_vals, alvo_q)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()

        soft_update(alvo, modelo, tau=TARGET_TAU)
        if hard_sync_step >= TARGET_SYNC_HARD:
            alvo.load_state_dict(modelo.state_dict())
            hard_sync_step = 0

    # === Autosave ===
    if global_step % AUTOSAVE_EVERY == 0:
        salvar_estado(modelo, otimizador, memoria, EPSILON, media_recompensa)

    # === Render estável ===
    env.render(campo)            # draw apenas
    pygame.display.flip()        # flip único global
    clock.tick(240)               # limite seguro de FPS

    # === Fim do episódio ===
    if terminado:
        recompensas.append(total_recompensa)
        media_recompensa = float(np.mean(recompensas)) if len(recompensas) else 0.0
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        taxa_poda = modelo.aplicar_poda(limiar_base=max(0.0005, 0.004 * (1 - EPSILON)))
        modelo.regenerar_sinapses(taxa_poda)

        try:
            modelo.verificar_homeostase(media_recompensa)
        except AttributeError:
            pass

        salvar_estado(modelo, otimizador, memoria, EPSILON, media_recompensa)

        print(
            f"🧬 Epoch {epoch:04d} | Reward={total_recompensa:7.1f} | Média={media_recompensa:7.1f} "
            f"| EPS={EPSILON:.3f} | Poda={taxa_poda*100:.2f}% | step={global_step}"
        )

        reseed()
        estado = env.reset()
        total_recompensa = 0.0
        epoch += 1
