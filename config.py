# ==========================================
# 🌌 Flappy EtherSym Turbo Trainer v9.5
# ==========================================
# Configuração projetada para máxima velocidade de treino:
# - Mínimo de renderização
# - Coleta massiva de experiências
# - Alta taxa de aprendizado + replay gigante
# - Ideal para GPU RTX / CUDA 11.8+
# ==========================================

# --- Mundo físico ---
LARGURA = 400
ALTURA = 600
CHAO = ALTURA - 80              # base visual simbólica

# --- Gravidade simbiótica (alta intensidade) ---
GRAVIDADE_BASE = 2.4
OSCILACAO_FREQUENCIA = 1.8
OSCILACAO_AMPLITUDE = 0.9

# --- Dinâmica do mundo (mais estímulos/segundo) ---
VELOCIDADE_CANO_BASE = 9.0      # acelera fluxo de obstáculos
DISTANCIA_CANO_BASE = 340       # reduz espaçamento (mais decisões)
GAP_VERTICAL_MIN = 150          # menor gap → mais aprendizado por erro
GAP_VERTICAL_MAX = 230
VELOCIDADE_PULO = -11.0
VELOCIDADE_DESCIDA = 8.5

# ==========================================
# 🤖 Treinamento por Reforço
# ==========================================

EPOCHS = 10000
BATCH = 128                     # mais amostras por atualização
GAMMA = 0.99                    # fator de desconto padrão
EPSILON_INICIAL = 1.0
EPSILON_DECAY = 0.9982          # decai mais rápido (mais exploração no início)
EPSILON_MIN = 0.05
LR = 0.0009                     # taxa de aprendizado ligeiramente maior

# Target Network (estabilidade)
TARGET_TAU = 0.015              # taxa de sincronização suave
TARGET_SYNC_HARD = 2500         # sincronização completa periódica

# ==========================================
# 💾 Replay Buffer
# ==========================================
MEMORIA_MAX = 500_000           # experiência massiva (RAM/GPU permitting)
MIN_REPLAY  = 8000              # precisa de buffer inicial mínimo
N_STEP = 5                      # maior profundidade temporal (melhor crédito)
# 💡 com 5-step bootstrapping o agente aprende padrões de longo prazo

# ==========================================
# ⚙️ Execução e Coleta
# ==========================================
FAST_MODE        = True         # desativa delays visuais
STEPS_PER_RENDER = 32           # executa 32 steps antes de atualizar tela
ACTION_REPEAT    = 6            # mantém mesma ação por 6 frames (menos overhead)
LOG_INTERVAL     = 2000         # imprime logs mais raramente (reduz I/O)
AUTOSAVE_EVERY   = 30000        # salva estado a cada 30k iterações (mínimo impacto)

# ==========================================
# 🖥️ Renderização (opcional)
# ==========================================
RENDER_INTERVAL = 0             # 0 = sem renderização (modo turbo)
FPS = 0                         # sem limitação de frames
# 💡 Pode ativar visualização temporária ajustando RENDER_INTERVAL=1000

# ==========================================
# 📂 Caminhos
# ==========================================
SAVE_PATH = "estado_treinamento.pth"
