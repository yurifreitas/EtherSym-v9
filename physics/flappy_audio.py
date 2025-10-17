import pygame
import numpy as np
import threading
import time
import random

# Inicializa mixer uma única vez
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

def gerar_onda_fluid(freq, duracao=1.5, volume=0.25):
    """Gera onda senoidal contínua e fluida com harmônicos e stereo vivo."""
    sr = 44100
    t = np.linspace(0, duracao, int(sr * duracao), endpoint=False)

    # harmônicos leves e defasados
    base = np.sin(2 * np.pi * freq * t)
    harm1 = np.sin(2 * np.pi * freq * 2 * t + np.pi / 3) * 0.3
    harm2 = np.sin(2 * np.pi * freq * 3 * t + np.pi / 5) * 0.15

    # envelope e respiro
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 1.7
    onda = (base + harm1 + harm2) * env * volume

    # estéreo fluido
    left = onda * (0.8 + 0.2 * np.sin(t * 2))
    right = onda * (0.8 + 0.2 * np.cos(t * 3))
    stereo = np.column_stack((left, right))
    return np.int16(stereo * 32767)

def gerar_som(freq, duracao, volume):
    """Cria o objeto pygame Sound a partir da onda."""
    arr = gerar_onda_fluid(freq, duracao, volume)
    return pygame.sndarray.make_sound(arr.copy())

def trilha_simbio_fluida(get_energia_callback):
    """Thread musical contínua — toca infinitamente, reagindo à energia."""
    escala = [0, 2, 4, 7, 9, 11, 12]
    base = 220.0  # A3
    notas = [base * (2 ** (n / 12)) for n in escala]

    # Pré-gera dois sons e sobrepõe continuamente (crossfade)
    som_atual = None
    while True:
        energia = np.clip(get_energia_callback(), 0.0, 1.0)
        freq = random.choice(notas) * (1 + random.uniform(-0.08, 0.08) * energia)
        duracao = 1.0 + energia * 0.8
        volume = 0.25 + energia * 0.3

        som = gerar_som(freq, duracao, volume)
        som.play(fade_ms=200)  # fade-in suave
        time.sleep(duracao * 0.75)  # inicia próxima nota antes da anterior terminar
