## 🌍 Idiomas / Languages / Idiomas

- 🇪🇸 [Español](README.es.md)
- 🇧🇷 [Português](README.pt.md)
- 🇺🇸 [English](README.en.md)


Extra:

 Función | Fórmula | Rango | Derivada | Notas |
|---------|---------|-------|----------|-------|
| ReLU | `ReLU(z) = max(0, z)` | [0, ∞) | 1 si z>0, 0 si z≤0 | Muy usada, evita saturación de gradiente |
| Sigmoide | `σ(z) = 1 / (1 + exp(-z))` | (0,1) | `σ(z)*(1-σ(z))` | Puede saturar, hoy se usa menos en capas ocultas |
| Tanh | `tanh(z) = (exp(z)-exp(-z))/(exp(z)+exp(-z))` | (-1,1) | `1 - tanh(z)^2` | Normaliza valores centrados en 0 |
