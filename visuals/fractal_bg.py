import numpy as np
import pygame
import math
import random


class FractalBackground:
    """Fundo fractal hipnótico leve e rápido — Mandelbrot e Julia híbrido com pulsação energética."""

    def __init__(self, largura: int, altura: int):
        self.largura = largura
        self.altura = altura
        self.modo = "mandelbrot"
        self.timer_modo = 0.0
        self.tempo_troca = 10.0
        self.c = complex(-0.7, 0.27015)
        self.zoom = 1.4
        self._frame_skip = 0
        self.buffer_surface = pygame.Surface((largura // 3, altura // 3))

    def _mandelbrot_fast(self, X, Y):
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.full(C.shape, True, dtype=bool)
        N = np.zeros(C.shape, dtype=np.float32)
        for i in range(40):
            Z[M] = Z[M] ** 2 + C[M]
            escaped = np.abs(Z) > 2
            N[escaped & M] = i
            M &= ~escaped
            if not M.any():
                break
        return N

    def _julia_fast(self, X, Y, c):
        Z = X + 1j * Y
        N = np.zeros(Z.shape, dtype=np.float32)
        M = np.full(Z.shape, True, dtype=bool)
        for i in range(40):
            Z[M] = Z[M] ** 2 + c
            escaped = np.abs(Z) > 2
            N[escaped & M] = i
            M &= ~escaped
            if not M.any():
                break
        return N

    def render(self, surface, energia_media: float, tempo: float):
        self._frame_skip = (self._frame_skip + 1) % 5
        if self._frame_skip != 0:
            surface.blit(
                pygame.transform.smoothscale(self.buffer_surface, (self.largura, self.altura)),
                (0, 0)
            )
            return

        if tempo - self.timer_modo > self.tempo_troca:
            self.modo = "julia" if self.modo == "mandelbrot" else "mandelbrot"
            self.timer_modo = tempo
            self.c = complex(random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8))

        small_w, small_h = self.largura // 3, self.altura // 3
        x = np.linspace(-2.0, 2.0, small_w)
        y = np.linspace(-1.5, 1.5, small_h)
        X, Y = np.meshgrid(x, y)

        self.zoom = 1.3 + math.sin(tempo * 0.3) * 0.25
        if self.modo == "mandelbrot":
            N = self._mandelbrot_fast(X / self.zoom, Y / self.zoom)
        else:
            N = self._julia_fast(X / self.zoom, Y / self.zoom, self.c)

        N = np.clip(N / 40.0, 0, 1.0)
        cor_base = np.array([
            60 + energia_media * 120,
            90 + energia_media * 150,
            180 + energia_media * 100
        ])

        rgb = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        rgb[..., 0] = ((N * 255 + cor_base[0]) % 255).astype(np.uint8)
        rgb[..., 1] = ((N * 180 + cor_base[1]) % 255).astype(np.uint8)
        rgb[..., 2] = ((N * 120 + cor_base[2]) % 255).astype(np.uint8)

        arr = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.blit_array(self.buffer_surface, arr)
        surface.blit(
            pygame.transform.smoothscale(self.buffer_surface, (self.largura, self.altura)),
            (0, 0)
        )
