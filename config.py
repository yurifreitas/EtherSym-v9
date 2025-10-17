# ==========================================
# 🌌 Flappy EtherSym — Turbo Trainer v9.7 (modo legado ultra-rápido)
# ==========================================

LARGURA = 400
ALTURA = 600
CHAO = ALTURA - 80

# --- Gravidade simbiótica intensa ---
GRAVIDADE_BASE = 2.4
OSCILACAO_FREQUENCIA = 1.8
OSCILACAO_AMPLITUDE = 0.9

# --- Dinâmica rápida ---
VELOCIDADE_CANO_BASE = 7.5
DISTANCIA_CANO_BASE = 380
GAP_VERTICAL_MIN = 180
GAP_VERTICAL_MAX = 240
VELOCIDADE_PULO = -10.5
VELOCIDADE_DESCIDA = 8.0

# ==========================================
# 🤖 Reforço
# ==========================================
EPOCHS = 10000
BATCH = 64
GAMMA = 0.99
EPSILON_INICIAL = 1.0
EPSILON_DECAY = 0.9985
EPSILON_MIN = 0.05
LR = 0.0007

TARGET_TAU = 0.02
TARGET_SYNC_HARD = 2000

# ==========================================
# 💾 Replay
# ==========================================
MEMORIA_MAX = 200_000
MIN_REPLAY  = max(BATCH * 5, 4000)
N_STEP = 3

# ==========================================
# ⚙️ Execução rápida
# ==========================================
# ==========================================
# ⚙️ Execução rápida
# ==========================================
FAST_MODE        = True
STEPS_PER_RENDER = 8
EPSILON_MIN = 0.08          # 🔁 permite leve exploração contínua
ACTION_REPEAT = 5            # 🔧 reduz sobrecarga de decisões
LOG_INTERVAL = 200           # 🧭 frequência de logs no terminal
AUTOSAVE_EVERY = 15000       # 💾 menos I/O, mais GPU
RENDER_INTERVAL = 0          # 🚀 modo turbo total (sem render)
FPS = 0


# ==========================================
# 📂 Caminhos
# ==========================================
SAVE_PATH = "estado_treinamento.pth"
