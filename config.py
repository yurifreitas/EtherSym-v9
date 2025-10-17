# ==========================================
# 🌌 Flappy EtherSym Turbo Trainer v9.6 — Estável & Evolutivo
# ==========================================
# Compatível com: main_train_evolutivo.py / main_watch.py
# Nunca precisa mudar entre versões — o sistema se adapta.
# ==========================================

# --- Mundo físico ---
LARGURA = 400
ALTURA = 600
CHAO = ALTURA - 80              # base simbólica do chão (visual)

# --- Gravidade simbiótica ---
GRAVIDADE_BASE = 2.4
OSCILACAO_FREQUENCIA = 1.8
OSCILACAO_AMPLITUDE = 0.9

# --- Dinâmica ---
VELOCIDADE_CANO_BASE = 9.0
DISTANCIA_CANO_BASE = 340
GAP_VERTICAL_MIN = 150
GAP_VERTICAL_MAX = 230
VELOCIDADE_PULO = -11.0
VELOCIDADE_DESCIDA = 8.5

# ==========================================
# 🤖 Treinamento por Reforço
# ==========================================
EPOCHS = 10000
BATCH = 128
GAMMA = 0.99
EPSILON_INICIAL = 1.0
EPSILON_DECAY = 0.9982
EPSILON_MIN = 0.05
LR = 0.0009

# Target Network
TARGET_TAU = 0.015
TARGET_SYNC_HARD = 2500

# ==========================================
# 💾 Replay Buffer
# ==========================================
MEMORIA_MAX = 500_000
MIN_REPLAY  = 8000
N_STEP = 5  # aprendizado de longo prazo

# ==========================================
# ⚙️ Execução e Coleta
# ==========================================
FAST_MODE        = True
STEPS_PER_RENDER = 32
ACTION_REPEAT    = 6
LOG_INTERVAL     = 2000
AUTOSAVE_EVERY   = 30000

# ==========================================
# 🖥️ Renderização (visualização)
# ==========================================
RENDER_INTERVAL = 0   # 0 = modo turbo (sem render)
FPS = 0               # 0 = ilimitado
# 💡 Defina RENDER_INTERVAL=1000 para visualizar enquanto treina

# ==========================================
# 🌱 Modo Evolutivo
# ==========================================
EVOLUTIVO = True          # True = continua sempre do último estado
CHECKPOINT_VERSIONS = 10  # quantos checkpoints manter antes de apagar os antigos

# ==========================================
# 📂 Caminhos
# ==========================================
SAVE_PATH = "estado_treinamento.pth"

# ==========================================
# 🧩 Compatibilidade garantida
# ==========================================
# Estes campos são usados pelos módulos de memória e rede.
# Nunca altere seus nomes; apenas ajuste valores se necessário.
