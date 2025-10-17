# ==========================================
# 🌌 Configurações principais do Flappy EtherSym
# (versão otimizada — aprendizado e progressão acelerados)
# ==========================================

# --- Mundo físico ---
LARGURA = 400          # largura da janela
ALTURA = 600           # altura total
CHAO = ALTURA - 80     # base simbólica do chão (deixa 80px visuais)

# --- Gravidade simbiótica ---
# Acelera resposta gravitacional para aprendizado mais intenso
GRAVIDADE_BASE = 2.4   
OSCILACAO_FREQUENCIA = 1.8
OSCILACAO_AMPLITUDE = 0.9

# --- Dimensões e velocidades ---
# Canos e movimentação mais rápidos → mais estímulos por segundo
VELOCIDADE_CANO_BASE = 7.5     # antes 5.0
DISTANCIA_CANO_BASE = 380      # antes 450
GAP_VERTICAL_MIN = 180         # aberturas menores exigem mais precisão
GAP_VERTICAL_MAX = 240
VELOCIDADE_PULO = -10.5
VELOCIDADE_DESCIDA = 8.0

# ==========================================
# 🤖 Treinamento de Aprendizado por Reforço
# ==========================================

EPOCHS = 10000                 # total de ciclos de treinamento
BATCH = 64                     # tamanho do lote
GAMMA = 0.99                   # fator de desconto
EPSILON_INICIAL = 1.0          # probabilidade inicial de exploração
EPSILON_DECAY = 0.9985         # antes 0.9994 → reduz exploração mais rápido
EPSILON_MIN = 0.05             # limite inferior
LR = 0.0007                    # taxa de aprendizado mais alta (antes 0.0004)

# ==========================================
# 🧬 Memória e Estado
# ==========================================

MEMORIA_MAX = 20000            # tamanho máximo do buffer de replay
SAVE_PATH = "estado_treinamento.pth"

# ==========================================
# 🖥️ Renderização e Loop Principal
# ==========================================

# Renderiza com maior frequência (menor delay visual)
RENDER_INTERVAL = 20           # antes 50
FPS = 600                      # antes 120 → acelera o loop principal
