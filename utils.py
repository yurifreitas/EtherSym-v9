import random, numpy as np, torch, time

def reseed():
    seed = int(time.time_ns() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def warmup(env, campo, steps=30):
    estado = env.reset()
    for _ in range(steps):
        acao = random.randint(0, 1)
        novo_estado, _, terminado = env.step(acao, campo)
        if terminado:
            return env.reset()
        estado = novo_estado
    return estado
