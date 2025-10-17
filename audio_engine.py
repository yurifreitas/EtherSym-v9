import pygame
import numpy as np
import random
import threading
import time

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Escalas simbióticas (modo lídio e frígio misturados)
ESCALA_BASE = [0, 2, 4, 6, 7, 9, 11, 12]
NOTAS = [220.0 * (2 ** (n / 12)) for n in ESCALA_BASE]  # base em A3

# Função para gerar um tom
def gerar_tom(freq, duracao=0.2, volume=0.3):
    amostra_rate = 44100
    t = np.linspace(0, duracao, int(amostra_rate * duracao), endpoint=False)
    onda = np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(t * 8))
    onda = (onda * 32767 * volume).astype(np.int16)
    stereo = np.column_stack((onda, onda))
    return pygame.sndarray.make_sound(stereo.copy())

# Sequência infinita harmônica
def tocar_loop_fractal(get_energia_callback):
    while True:
        energia = get_energia_callback()
        # frequência e cadência mudam conforme energia
        base_freq = random.choice(NOTAS)
        freq = base_freq * (1 + energia * random.uniform(-0.15, 0.15))
        duracao = max(0.12, 0.5 - energia * 0.4)
        som = gerar_tom(freq, duracao, volume=0.25 + energia * 0.2)
        som.play()
        time.sleep(duracao * (0.9 + random.random() * 0.2))
