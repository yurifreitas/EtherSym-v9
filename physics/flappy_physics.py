import random, math, numpy as np, time
from config import *
from .flappy_utils import gerar_cano, salvar_max_score

def atualizar_movimento(env, acao, campo):
    p = env.passaro
    recompensa = 0.0
    agora = time.time()

    # === Gravidade e movimento ===
    g_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
    osc = math.sin(agora * 1.6 + random.random()) * OSCILACAO_AMPLITUDE
    gravidade = (env.gravidade_base * math.copysign(1, g_local)) + (g_local * 2.0) + osc
    p["vel"] += gravidade
    p["vel"] = float(np.clip(p["vel"], -12.0, 12.0))
    p["y"] += p["vel"]

    # === Ações ===
    if acao == 1 and p["y"] > 40:
        p["vel"] = VELOCIDADE_PULO
        p["energia"] -= 0.025
    elif acao == -1 and p["y"] < CHAO - 40:
        p["vel"] = VELOCIDADE_DESCIDA
        p["energia"] -= 0.02
    else:
        p["energia"] -= 0.004

    # === Campo simbiótico ===
    campo.evolve(p["energia"], random.uniform(-0.25, 0.25))
    energia_local = campo.gravidade_local(p["x"], p["y"], LARGURA, ALTURA)
    p["energia"] = np.clip(p["energia"] + energia_local * 0.03, 0.0, 1.0)
    recompensa += energia_local * 2.0

    # === Atualiza canos ===
    for cano in env.canos:
        cano["x"] -= VELOCIDADE_CANO_BASE

    # Gera novos canos se necessário
    if not env.canos or env.canos[-1]["x"] < LARGURA - DISTANCIA_CANO_BASE:
        env.canos.append(gerar_cano(env.canos))

    # Remove antigos
    env.canos = [c for c in env.canos if c["x"] > -120]

    # === Pontuação ===
    for cano in env.canos:
        if not cano["scored"] and cano["x"] + 70 < p["x"]:
            cano["scored"] = True
            env.pontuacao += 1
            env.max_score = max(env.max_score, env.pontuacao)
            salvar_max_score(env.max_score_path, env.max_score)
            recompensa += 25

    # === Colisões ===
    cano_colidido, morreu = checar_colisao(env)
    if morreu:
        recompensa -= 300
    p["y"] = np.clip(p["y"], 0.0, CHAO - 1)

    return env._get_estado(acao), recompensa, cano_colidido, not morreu

def checar_colisao(env):
    p = env.passaro
    for cano in env.canos:
        cano_x, cano_h = cano["x"], cano["altura"]
        raio = 15
        passou_gap = (cano_h - 90 - raio) < p["y"] < (cano_h + 90 + raio)
        dentro_x = (cano_x - 30) < p["x"] < (cano_x + 100)
        if dentro_x and not passou_gap:
            return cano, True
    if p["y"] > (CHAO - 3) or p["y"] < 3:
        return None, True
    return None, False
