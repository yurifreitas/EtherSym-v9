import random, math, numpy as np, time
from config import *
from .flappy_utils import gerar_cano, salvar_max_score

# =======================================================
# 🌌 Ambiente simbiótico extremo — EtherSym: Caos Total
# =======================================================

def atualizar_movimento(env, acao, campo):
    p = env.passaro
    recompensa = 0.0
    agora = time.time()

    # --- VARIAÇÃO CONTEXTUAL DE UNIVERSO ---
    if not hasattr(env, "assinatura"):
        env.assinatura = random.random() * math.pi * 2
    variacao = math.sin(agora * 0.4 + env.assinatura) * random.uniform(0.6, 1.4)
    env.gravidade_base = GRAVIDADE_BASE * (1 + 0.8 * variacao)

    # --- GRAVIDADE SIMBIÓTICA COMPLEXA ---
    base = env.gravidade_base
    g_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
    osc_primaria = math.sin(agora * random.uniform(1.2, 2.5)) * OSCILACAO_AMPLITUDE
    osc_secundaria = math.cos(agora * 0.9 + random.random()) * 1.1
    inversao = math.copysign(1, math.sin(agora * 0.07 + env.assinatura))
    caos_temporal = math.sin(agora * random.uniform(0.05, 0.2)) * random.uniform(-2.5, 2.5)
    turbulencia = random.uniform(-0.8, 0.8)

    gravidade = (base * inversao) + g_local * 1.8 + osc_primaria + osc_secundaria + caos_temporal + turbulencia
    p["vel"] += gravidade
    p["vel"] = float(np.clip(p["vel"], -14.0, 14.0))
    p["y"] += p["vel"]

    # --- VENTO QUÂNTICO (caótico e mutável) ---
    vento = (
        math.sin(agora * 1.1 + p["y"] * 0.012)
        + math.cos(agora * 0.5 + p["x"] * 0.015)
        + random.uniform(-0.4, 0.4)
    ) * random.uniform(0.5, 1.4)
    p["x"] += vento * 0.25

    # --- ATRASO TEMPORAL (delay simbiótico) ---
    if not hasattr(env, "buffer_acoes"):
        env.buffer_acoes = [0, 0, 0, 0]
    env.buffer_acoes.append(acao)
    acao_efetiva = env.buffer_acoes.pop(0)

    # --- AÇÕES COM FADIGA ---
    fadiga = random.uniform(0.9, 1.2)
    if acao_efetiva == 1 and p["y"] > 40:
        p["vel"] = VELOCIDADE_PULO * fadiga
        p["energia"] -= 0.035
    elif acao_efetiva == -1 and p["y"] < CHAO - 40:
        p["vel"] = VELOCIDADE_DESCIDA * fadiga
        p["energia"] -= 0.03
    else:
        p["energia"] -= 0.0045

    # --- CAMPO SIMBIÓTICO ---
    campo.evolve(p["energia"], random.uniform(-0.35, 0.35))
    energia_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
    p["energia"] = np.clip(p["energia"] + energia_local * 0.05, 0.0, 1.0)
    recompensa += energia_local * random.uniform(2.0, 3.5)

    # --- CANOS EM MUTAÇÃO ---
    for cano in env.canos:
        if "fase" not in cano:
            cano["fase"] = random.random() * math.pi * 2
            cano["vel_var"] = random.uniform(0.9, 1.3)
        cano["x"] -= VELOCIDADE_CANO_BASE * cano["vel_var"]
        cano["altura"] += math.sin(agora * random.uniform(1.3, 1.8) + cano["fase"]) * random.uniform(0.7, 1.3)

    # Gera novos canos com gaps e espaçamentos aleatórios
    if not env.canos or env.canos[-1]["x"] < LARGURA - random.randint(320, 460):
        env.canos.append(gerar_cano(env.canos))

    # Remove antigos
    env.canos = [c for c in env.canos if c["x"] > -120]

    # --- ZONAS DE SOMBRA (perda de percepção) ---
    if random.random() < 0.005:
        env.blur_ativo = not getattr(env, "blur_ativo", False)
    if getattr(env, "blur_ativo", False):
        recompensa -= 0.5  # penaliza por “baixa visibilidade”

    # --- PONTUAÇÃO ---
    for cano in env.canos:
        if not cano.get("scored") and cano["x"] + 70 < p["x"]:
            cano["scored"] = True
            env.pontuacao += 1
            env.max_score = max(env.max_score, env.pontuacao)
            salvar_max_score(env.max_score_path, env.max_score)
            recompensa += 25 * random.uniform(0.8, 1.2)

    # --- RECOMPENSA ANTIAPEGADA ---
    if int(agora * 0.5) % 9 == 0:
        recompensa = -recompensa * random.uniform(0.8, 1.3)

    # --- COLISÕES ---
    cano_colidido, morreu = checar_colisao(env, vento)
    if morreu:
        recompensa -= 350 + random.uniform(0, 100)
    p["y"] = np.clip(p["y"], 0.0, CHAO - 1)

    # --- DIFICULDADE PROGRESSIVA ---
    if env.pontuacao and env.pontuacao % 10 == 0:
        env.gravidade_base += random.uniform(0.1, 0.3)
        for cano in env.canos:
            if "vel_var" not in cano:
                cano["vel_var"] = random.uniform(0.9, 1.3)
            cano["vel_var"] *= random.uniform(1.05, 1.15)


    # --- ESTADO RUIDOSO (percepção imperfeita) ---
    estado_real = env._get_estado(acao)
    ruido = np.random.normal(0, 0.02 * random.uniform(0.5, 1.5), size=estado_real.shape)
    estado_ruidoso = np.clip(estado_real + ruido, -1, 1)

    return estado_ruidoso, recompensa, cano_colidido, not morreu


def checar_colisao(env, vento=0.0):
    p = env.passaro
    for cano in env.canos:
        cano_x, cano_h = cano["x"], cano["altura"]
        raio = 15
        gap = random.randint(70, 100)
        passou_gap = (cano_h - gap - raio) < p["y"] < (cano_h + gap + raio)
        dentro_x = (cano_x - 30 + vento) < p["x"] < (cano_x + 100 + vento)
        if dentro_x and not passou_gap:
            return cano, True
    if p["y"] > (CHAO - 3) or p["y"] < 3:
        return None, True
    return None, False
