## 🌍 Idiomas / Languages / Idiomas

- 🇪🇸 [Español](README.es.md)
- 🇧🇷 [Português](README.pt.md)
- 🇺🇸 [English](README.en.md)


Extra:

# Guía rápida de funciones de activación y pérdida en redes neuronales

Esta guía resume **qué función de activación y pérdida usar según tu tarea**, y las fórmulas más importantes para referencia rápida.

---

## 1. Activaciones en capas ocultas

| Función | Fórmula | Rango | Derivada | Notas |
|---------|---------|-------|----------|-------|
| ReLU | `ReLU(z) = max(0, z)` | [0, ∞) | 1 si z>0, 0 si z≤0 | Muy usada, evita saturación de gradiente |
| Sigmoide | `σ(z) = 1 / (1 + exp(-z))` | (0,1) | `σ(z)*(1-σ(z))` | Puede saturar, hoy se usa menos en capas ocultas |
| Tanh | `tanh(z) = (exp(z)-exp(-z))/(exp(z)+exp(-z))` | (-1,1) | `1 - tanh(z)^2` | Normaliza valores centrados en 0 |

**💡 Regla práctica:** casi siempre se usa **ReLU** en capas ocultas.

---

## 2. Activación en capa de salida según la tarea

| Tarea | Activación | Rango | Pérdida típica | Notas |
|-------|------------|-------|----------------|-------|
| Clasificación binaria (0/1) | Sigmoide | (0,1) | Binary Cross-Entropy (BCE) | Predice probabilidad de clase positiva |
| Clasificación multiclase (una clase correcta) | Softmax | (0,1), suma=1 | Categorical Cross-Entropy | Cada neurona = clase, salida interpretable como probabilidad |
| Clasificación multietiqueta | Sigmoide | (0,1) por neurona | Binary Cross-Entropy por neurona | Cada clase independiente (ej: “gato” y “perro”) |
| Regresión (valores continuos) | Lineal (sin activación) | (-∞, ∞) | MSE o MAE | Predicción directa de valores |
| Regresión con salida positiva | ReLU | [0, ∞) | MSE o MAE | Evita predicciones negativas |

---

## 3. Fórmulas de activación y derivadas

- **Sigmoide:**  
\[
\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z) (1-\sigma(z))
\]

- **ReLU:**  
\[
\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = 
\begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}
\]

- **Tanh:**  
\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)
\]

- **Softmax (vectorial):**  
\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]  
> Para cross-entropy, el gradiente se simplifica a:  
\[
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
\]

---

## 4. Resumen rápido (diagrama mental)

1. **Capa oculta** → ReLU  
2. **Capa salida**:
   - 0/1 → Sigmoide + BCE  
   - Multiclase → Softmax + Categorical Cross-Entropy  
   - Regresión → Lineal (MSE/MAE)  
   - Regresión ≥0 → ReLU (MSE/MAE)  

---

Con esta guía puedes decidir rápido qué activación y pérdida usar según tu proyecto.
