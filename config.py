# ==========================================
# 🌌 Configurações principais do Flappy EtherSym
# (versão otimizada — aprendizado e progressão acelerados)
# ==========================================

# --- Mundo físico ---
LARGURA = 400
ALTURA = 600
CHAO = ALTURA - 80

# --- Gravidade simbiótica (mais intensa) ---
GRAVIDADE_BASE = 2.4
OSCILACAO_FREQUENCIA = 1.8
OSCILACAO_AMPLITUDE = 0.9

# --- Dimensões e velocidades (mais estímulos/seg) ---
VELOCIDADE_CANO_BASE = 7.5      # antes 5.0
DISTANCIA_CANO_BASE = 380       # antes 450
GAP_VERTICAL_MIN = 180
GAP_VERTICAL_MAX = 240
VELOCIDADE_PULO = -10.5
VELOCIDADE_DESCIDA = 8.0

# ==========================================
# 🤖 Treinamento de Aprendizado por Reforço
# ==========================================
EPOCHS = 10000
BATCH = 64
GAMMA = 0.99
EPSILON_INICIAL = 1.0
EPSILON_DECAY = 0.9985          # antes 0.9994
EPSILON_MIN = 0.05
LR = 0.0007

# Target network
TARGET_TAU = 0.02
TARGET_SYNC_HARD = 2000

# Replay
MEMORIA_MAX = 200_000           # ring buffer grande (acelera IO)
MIN_REPLAY  = max(BATCH * 5, 4000)

# N-step return
N_STEP = 3

# Execução rápida / coleta
FAST_MODE        = True
STEPS_PER_RENDER = 8            # passos de ambiente por ciclo de render
ACTION_REPEAT    = 4            # repete a mesma ação K steps
LOG_INTERVAL     = 200
AUTOSAVE_EVERY   = 10000        # espaçar I/O

# Renderização
RENDER_INTERVAL = 10            # antes 50
FPS = 60                        # antes 120

# Caminhos
SAVE_PATH = "estado_treinamento.pth"
