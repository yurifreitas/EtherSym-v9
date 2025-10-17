# ==========================================
# 🌌 EtherSym v9 — Turbo Dueling DQN Simbiótico (MAIN COMPLETO LIMPO)
# ==========================================

import os, sys, random, numpy as np, pygame, torch
from torch.amp import GradScaler, autocast  # PyTorch ≥ 2.6

from config import (
    LARGURA, ALTURA, FPS, FAST_MODE, RENDER_INTERVAL,
    STEPS_PER_RENDER, ACTION_REPEAT, LOG_INTERVAL, AUTOSAVE_EVERY,
    BATCH, GAMMA, EPSILON_INICIAL, EPSILON_DECAY, EPSILON_MIN,
    N_STEP, TARGET_TAU, TARGET_SYNC_HARD, SAVE_PATH, MIN_REPLAY,
    MEMORIA_MAX
)
from utils import warmup
from field import GravidadeAstrofisica
from physics.flappy_env import AmbienteFlappy
from memory import salvar_estado, carregar_estado
from network import criar_modelo
from replay import RingReplay, NStepBuffer

# ========================
# 🔧 Turbo CUDA / determinismo suave
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = "tf32"  # ✅ nova forma recomendada
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_num_threads(1)
except Exception:
    pass

def reseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🌱 RNG reseed com seed={seed}")

reseed(42)

# ========================
# 🪟 Inicialização da janela (headless opcional)
# ========================
if os.environ.get("HEADLESS", "0") == "1":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.font.init()

DISPLAY_FLAGS = pygame.DOUBLEBUF | pygame.HWSURFACE
try:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS, vsync=0)
except Exception:
    TELA = pygame.display.set_mode((LARGURA, ALTURA), DISPLAY_FLAGS)

pygame.display.set_caption("🌌 EtherSym v9 — Turbo Dueling DQN (Treino)")
clock = pygame.time.Clock()
pygame.event.set_allowed([pygame.QUIT])

# ========================
# 🧠 Modelo / alvo / opt / loss
# ========================
modelo, alvo, opt, loss_fn = criar_modelo(device)
alvo.eval()
modelo.train(True)

# Compilação adaptativa
try:
    if device.type == "cuda":
        modelo = torch.compile(modelo, mode="max-autotune")
        alvo   = torch.compile(alvo,   mode="max-autotune")
except Exception:
    pass

# GradScaler (nova API)
scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

# Carrega estado salvo (incremental)
_legacy_mem, EPSILON, media_recompensa = carregar_estado(modelo, opt)
if EPSILON is None:
    EPSILON = EPSILON_INICIAL
print(f"🌱 Evoluindo modelo existente | ε={EPSILON:.3f} | média_antiga={float(media_recompensa or 0):.2f}")

# ========================
# 🌌 Ambiente e buffers
# ========================
campo = GravidadeAstrofisica()
env   = AmbienteFlappy()

ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
def a_to_index(a): return int(a + 1)

replay = RingReplay(state_dim=6, capacity=MEMORIA_MAX, device=device)
nstep_helper = NStepBuffer(N_STEP, GAMMA)

# ========================
# 🔮 Política (ε-greedy + temperatura)
# ========================
TEMPERATURA_BASE = 0.95
TEMPERATURA_MIN  = 0.60

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

@torch.no_grad()
def soft_update(alvo_m, online_m, tau=TARGET_TAU):
    for p_t, p in zip(alvo_m.parameters(), online_m.parameters()):
        p_t.data.mul_(1.0 - tau).add_(tau * p.data)

def step_repetido(env, acao, campo, repeats=ACTION_REPEAT):
    total_r, s, done = 0.0, None, False
    for _ in range(max(1, repeats)):
        s, r, done = env.step(acao, campo)
        total_r += r
        if done:
            break
    return s, float(total_r), done

# ========================
# 🚀 Loop principal
# ========================
try:
    estado = warmup(env, campo)
except Exception:
    estado = env.reset()

total_recompensa = 0.0
epoch = 0
recompensas = []
global_step = 0
hard_sync_step = 0
running = True

print("🚀 Iniciando Treinamento Turbo Simbiótico...")

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # ===== Coleta turbo =====
    for _ in range(max(1, STEPS_PER_RENDER)):
        acao = escolher_acao(estado)
        novo_estado, recompensa, terminado = step_repetido(env, acao, campo, repeats=ACTION_REPEAT)
        recompensa = float(np.clip(recompensa, -300.0, 50.0))

        nstep_helper.push(estado, acao, recompensa)
        flush = (len(nstep_helper.traj) == N_STEP) or terminado
        if flush:
            item = nstep_helper.flush(novo_estado, terminado)
            if item is not None:
                s0, a0, Rn, s_n, done_n = item
                replay.append(s0, a_to_index(a0), Rn, s_n, float(done_n))

        total_recompensa += recompensa
        estado = novo_estado
        global_step += 1
        hard_sync_step += 1

        # ===== Aprendizado =====
        if len(replay) >= MIN_REPLAY:
            estados_t, acoes_t, recompensas_t, novos_estados_t, finais_t = replay.sample(BATCH)

            with torch.no_grad():
                next_online_q = modelo(novos_estados_t)
                next_actions  = torch.argmax(next_online_q, dim=1)
                next_target_q = alvo(novos_estados_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                alvo_q        = recompensas_t + (GAMMA ** N_STEP) * next_target_q * (1.0 - finais_t)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                q_vals = modelo(estados_t).gather(1, acoes_t).squeeze(1)
                perda  = loss_fn(q_vals, alvo_q)

            scaler.scale(perda).backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            soft_update(alvo, modelo, tau=TARGET_TAU)
            if hard_sync_step >= TARGET_SYNC_HARD:
                alvo.load_state_dict(modelo.state_dict())
                hard_sync_step = 0

        # ===== Fim de episódio =====
        if terminado:
            recompensas.append(total_recompensa)
            media_recompensa = float(np.mean(recompensas[-120:])) if len(recompensas) else 0.0
            EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

            taxa_poda = 0.0
            if epoch % 2 == 0:
                try:
                    taxa_poda = modelo.aplicar_poda(limiar_base=max(0.0005, 0.004 * (1 - EPSILON)))
                    modelo.regenerar_sinapses(taxa_poda)
                except AttributeError:
                    pass
            try:
                modelo.verificar_homeostase(media_recompensa)
            except AttributeError:
                pass

            if epoch % max(1, LOG_INTERVAL // 5) == 0:
                print(
                    f"🧬 Epoch {epoch:04d} | R={total_recompensa:7.1f} | "
                    f"Média={media_recompensa:7.1f} | EPS={EPSILON:.3f} | "
                    f"steps={global_step} | Poda={taxa_poda*100:.2f}%"
                )

            salvar_estado(modelo, opt, [], EPSILON, media_recompensa)
            estado = env.reset()
            total_recompensa = 0.0
            epoch += 1

    # ===== Renderização =====
    if not FAST_MODE:
        if RENDER_INTERVAL <= 1:
            env.render(campo); pygame.display.flip(); clock.tick(FPS or 0)
        elif (global_step % RENDER_INTERVAL) == 0:
            env.render(campo); pygame.display.flip()
    else:
        if RENDER_INTERVAL > 0 and (global_step % RENDER_INTERVAL) == 0:
            env.render(campo); pygame.display.flip()

# ========================
# Encerramento
# ========================
try:
    TELA.fill((0, 0, 0)); pygame.display.flip()
except Exception:
    pass
salvar_estado(modelo, opt, [], EPSILON, locals().get("media_recompensa", 0.0))
pygame.time.delay(100)
pygame.quit()
sys.exit(0)
