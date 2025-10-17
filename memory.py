# ==========================================
# 🧠 EtherSym Memory System — v9.7.1 (Evolutivo + Autorreparador)
# ==========================================

import os, pickle, time, shutil, torch
from collections import deque
from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL

# ------------------------------------------
# 🔐 Salvamento seguro
# ------------------------------------------
def _safe_replace(tmp_path, final_path):
    """Substitui um arquivo de forma atômica e sincronizada (sem risco de corrupção)."""
    os.replace(tmp_path, final_path)
    if hasattr(os, "sync"):
        try:
            os.sync()
        except:
            pass


# ------------------------------------------
# 💾 Salvar estado principal
# ------------------------------------------
def salvar_estado(modelo, otimizador, memoria, epsilon, media_recompensa):
    """Salva o estado simbiótico completo (pesos, otimizador, epsilon, média, memória)."""
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
        lista = list(memoria)[-MEMORIA_MAX:] if memoria is not None else []
        pickle.dump(lista, f, protocol=pickle.HIGHEST_PROTOCOL)
    _safe_replace(tmp_mem, SAVE_PATH + ".mem")

    print(f"💾 Estado salvo: ε={epsilon:.3f} | média={media_recompensa:.2f} | memória={len(memoria) if memoria else 0}")


# ------------------------------------------
# 🧬 Salvar incremental (histórico evolutivo)
# ------------------------------------------
def salvar_estado_incremental(modelo, otimizador, memoria, epsilon, media_recompensa):
    """Cria uma cópia versionada do estado (checkpoint evolutivo)."""
    salvar_estado(modelo, otimizador, memoria, epsilon, media_recompensa)
    base, ext = os.path.splitext(SAVE_PATH)
    stamp = int(time.time())
    backup = f"{base}_v{stamp}{ext}"
    try:
        shutil.copy2(SAVE_PATH, backup)
        print(f"🧬 Checkpoint evolutivo criado: {backup}")
    except Exception as e:
        print(f"⚠️ Falha ao criar checkpoint evolutivo: {e}")


# ------------------------------------------
# 🔄 Carregar estado e regenerar se necessário
# ------------------------------------------
def carregar_estado(modelo, otimizador):
    """Carrega o estado simbiótico com tolerância a erros e regeneração automática."""
    memoria = deque(maxlen=MEMORIA_MAX)
    epsilon, media = EPSILON_INICIAL, 0.0

    if not os.path.exists(SAVE_PATH):
        print("🆕 Nenhum estado anterior — iniciando do zero (modo evolutivo ativo).")
        return memoria, epsilon, media

    try:
        data = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            if "modelo" in data:
                modelo.load_state_dict(data["modelo"], strict=False)
            if "otimizador" in data:
                otimizador.load_state_dict(data["otimizador"])
            epsilon = float(data.get("epsilon", EPSILON_INICIAL))
            media = float(data.get("media_recompensa", 0.0))
        else:
            modelo.load_state_dict(data, strict=False)

        # --- restaura memória simbiótica ---
        mem_path = SAVE_PATH + ".mem"
        if os.path.exists(mem_path):
            with open(mem_path, "rb") as f:
                mem_data = pickle.load(f)
                if isinstance(mem_data, list):
                    memoria.extend(mem_data[-MEMORIA_MAX:])

        # --- detecção simbiótica de reset (autorreparo) ---
        if len(memoria) == 0 and media < -500 and epsilon <= 0.06:
            print("⚡ Memória vazia detectada com baixa média — regenerando aprendizado simbiótico.")
            epsilon = 1.0  # reinicia exploração
            media = 0.0

        print(f"🧬 Estado restaurado: ε={epsilon:.3f} | média={media:.2f} | memória={len(memoria)}")

    except Exception as e:
        print(f"⚠️ Erro ao carregar estado ({type(e).__name__}): {e}")
        print("🔄 Reiniciando com pesos atuais e memória limpa (modo evolutivo segue).")
        memoria = deque(maxlen=MEMORIA_MAX)
        epsilon, media = EPSILON_INICIAL, 0.0

    return memoria, epsilon, media
