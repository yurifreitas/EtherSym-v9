import torch, pickle, os, tempfile
from collections import deque
from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL


def salvar_estado(modelo, otimizador, memoria, epsilon, media_recompensa):
    """Salva o estado simbiótico completo de forma atômica e segura."""
    data = {
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "epsilon": epsilon,
        "media_recompensa": media_recompensa,
    }

    # === Salvamento atômico do estado ===
    tmp_path = SAVE_PATH + ".tmp"
    torch.save(data, tmp_path, pickle_protocol=5)
    os.replace(tmp_path, SAVE_PATH)

    # === Salvamento da memória simbiótica ===
    tmp_mem = SAVE_PATH + ".mem.tmp"
    with open(tmp_mem, "wb") as f:
        # Limita tamanho para evitar explosão de RAM/disk
        pickle.dump(list(memoria)[-MEMORIA_MAX:], f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_mem, SAVE_PATH + ".mem")

    print(f"💾 Estado salvo: ε={epsilon:.3f} | média={media_recompensa:.2f} | memória={len(memoria)}")


def carregar_estado(modelo, otimizador):
    """Carrega o estado simbiótico com tolerância a erros e arquivos corrompidos."""
    memoria = deque(maxlen=MEMORIA_MAX)
    epsilon, media = EPSILON_INICIAL, 0.0

    if not os.path.exists(SAVE_PATH):
        print("🆕 Nenhum estado anterior — iniciando treinamento simbiótico do zero.")
        return memoria, epsilon, media

    try:
        data = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)

        if "modelo" in data:
            modelo.load_state_dict(data["modelo"], strict=False)
        if "otimizador" in data:
            otimizador.load_state_dict(data["otimizador"])
        epsilon = data.get("epsilon", EPSILON_INICIAL)
        media = data.get("media_recompensa", 0.0)

        # === Tenta carregar memória simbiótica separada ===
        mem_path = SAVE_PATH + ".mem"
        if os.path.exists(mem_path):
            with open(mem_path, "rb") as f:
                mem_data = pickle.load(f)
                if isinstance(mem_data, list):
                    memoria.extend(mem_data[-MEMORIA_MAX:])
        print(f"🧬 Estado restaurado: ε={epsilon:.3f} | média={media:.2f} | memória={len(memoria)}")

    except Exception as e:
        print(f"⚠️ Erro ao carregar estado ({type(e).__name__}): {e}")
        print("🔄 Reiniciando memória simbiótica limpa.")
        memoria = deque(maxlen=MEMORIA_MAX)
        epsilon, media = EPSILON_INICIAL, 0.0

    return memoria, epsilon, media
