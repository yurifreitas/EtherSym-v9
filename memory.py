import os, pickle, time, shutil
import torch
from collections import deque
from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL

def _safe_replace(tmp_path, final_path):
    os.replace(tmp_path, final_path)
    if hasattr(os, "sync"):
        try: os.sync()
        except: pass

def salvar_estado(modelo, otimizador, memoria, epsilon, media_recompensa):
    """Salva o estado completo de forma atômica (checkpoint principal)."""
    data = {
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "epsilon": float(epsilon),
        "media_recompensa": float(media_recompensa),
    }

    # — checkpoint .pth (atômico)
    tmp_path = SAVE_PATH + ".tmp"
    torch.save(data, tmp_path, pickle_protocol=5)
    _safe_replace(tmp_path, SAVE_PATH)

    # — memória simbiótica .mem (atômico)
    tmp_mem = SAVE_PATH + ".mem.tmp"
    with open(tmp_mem, "wb") as f:
        # garante tamanho máx
        lista = list(memoria)[-MEMORIA_MAX:] if memoria is not None else []
        pickle.dump(lista, f, protocol=pickle.HIGHEST_PROTOCOL)
    _safe_replace(tmp_mem, SAVE_PATH + ".mem")

    print(f"💾 Estado salvo: ε={epsilon:.3f} | média={media_recompensa:.2f} | memória={len(memoria) if memoria else 0}")

def salvar_estado_incremental(modelo, otimizador, memoria, epsilon, media_recompensa):
    """Salva o estado + cria uma cópia versionada para histórico evolutivo."""
    salvar_estado(modelo, otimizador, memoria, epsilon, media_recompensa)
    base, ext = os.path.splitext(SAVE_PATH)
    stamp = int(time.time())
    backup = f"{base}_v{stamp}{ext}"
    try:
        shutil.copy2(SAVE_PATH, backup)
        print(f"🧬 Checkpoint evolutivo criado: {backup}")
    except Exception as e:
        print(f"⚠️ Falha ao criar checkpoint evolutivo: {e}")

def carregar_estado(modelo, otimizador):
    """Carrega o estado com tolerância a erros. Mescla memória existente."""
    from collections import deque
    memoria = deque(maxlen=MEMORIA_MAX)
    epsilon, media = EPSILON_INICIAL, 0.0

    if not os.path.exists(SAVE_PATH):
        print("🆕 Nenhum estado anterior — iniciando do zero (evolutivo habilitado).")
        return memoria, epsilon, media

    try:
        # PyTorch 2.6+: weights_only=False se o .pth tiver mais que pesos
        data = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            if "modelo" in data:
                modelo.load_state_dict(data["modelo"], strict=False)
            if "otimizador" in data:
                otimizador.load_state_dict(data["otimizador"])
            epsilon = float(data.get("epsilon", EPSILON_INICIAL))
            media   = float(data.get("media_recompensa", 0.0))
        else:
            # caso seja state_dict puro
            modelo.load_state_dict(data, strict=False)

        # — mescla memória simbiótica (se existir)
        mem_path = SAVE_PATH + ".mem"
        if os.path.exists(mem_path):
            with open(mem_path, "rb") as f:
                mem_data = pickle.load(f)
                if isinstance(mem_data, list):
                    memoria.extend(mem_data[-MEMORIA_MAX:])

        print(f"🧬 Estado restaurado: ε={epsilon:.3f} | média={media:.2f} | memória={len(memoria)}")

    except Exception as e:
        print(f"⚠️ Erro ao carregar estado ({type(e).__name__}): {e}")
        print("🔄 Iniciando com pesos atuais e memória vazia (modo evolutivo segue).")
        memoria = deque(maxlen=MEMORIA_MAX)
        epsilon, media = EPSILON_INICIAL, 0.0

    return memoria, epsilon, media
