import json, random
from config import *

def gerar_cano(canos):
    ultimo_x = canos[-1]["x"] if canos else 0
    distancia = DISTANCIA_CANO_BASE + random.randint(50, 100)
    gap = random.randint(GAP_VERTICAL_MIN, GAP_VERTICAL_MAX)
    altura = random.randint(180, 420)
    return {"x": ultimo_x + distancia, "altura": altura, "scored": False}

def carregar_max_score(path):
    try:
        with open(path, "r") as f:
            return json.load(f).get("max_score", 0)
    except Exception:
        return 0

def salvar_max_score(path, score):
    try:
        with open(path, "w") as f:
            json.dump({"max_score": score}, f)
    except Exception:
        pass
