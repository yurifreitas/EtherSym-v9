import torch
import numpy as np

class GravidadeAstrofisica:
    """
    Campo simbiótico de gravidade — reage à energia do agente.
    Baseia-se em um mapa tensorial que se deforma conforme o voo e a energia.
    """
    def __init__(self, largura=20, altura=15):
        self.size = (largura, altura)
        self.field = torch.zeros(self.size, dtype=torch.float32)
        self.noise = torch.randn(self.size, dtype=torch.float32)
        self.alpha = 0.97  # fator de amortecimento
        self.beta = 0.05   # fator de excitação

    def evolve(self, energia, turbulencia=0.5):
        """
        Atualiza o campo gravitacional de forma simbiótica.
        - energia: nível de energia do pássaro
        - turbulência: ruído caótico leve
        """
        ruido = (torch.randn_like(self.field) * turbulencia * 0.02)
        self.field = (
            self.alpha * self.field
            + self.beta * self.noise * energia
            + ruido
        )
        self.field = torch.clamp(self.field, -1.0, 1.0)

        # deforma o ruído base lentamente (movimento do "éter")
        self.noise = torch.roll(self.noise, shifts=(1, -1), dims=(0, 1))

    def gravidade_local(self, x, y, largura_tela, altura_tela):
        """
        Retorna o valor local do campo gravitacional no ponto (x, y)
        mapeado para o tensor interno.
        """
        i = int((x / largura_tela) * (self.size[0] - 1))
        j = int((y / altura_tela) * (self.size[1] - 1))
        i = np.clip(i, 0, self.size[0] - 1)
        j = np.clip(j, 0, self.size[1] - 1)
        return float(self.field[j, i])
