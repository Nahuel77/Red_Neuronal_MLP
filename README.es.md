Primeramente, aclarar que más que especificaciones técnicas, dejaré aquí algunas consideraciones más cercanas a lo que involucran las nociones básicas de su funcionamiento.
A modo de recordar lo aprendido, aquí paso a explicarme a mí mismo y a quien visite este repositorio.

Se ha dicho a modo de chiste que la IA es un nido de bucles "IF".
Dentro de las IA se podría decir, sin tanto chiste, que la red neuronal diseñada en este repositorio es un proceso de estadísticas aplicadas a matrices (o, para Python, arrays bidimensionales*) de un modo secuencial, obteniendo así una última matriz de capa, donde el valor más alto es el resultado esperado.

Básicamente, lo que aquí se codifica es un grupo de matrices que se interrelacionan, pero que, al momento de entrenarlas, se les da las respuestas para que luego se ajusten esas relaciones.

Esta red neuronal pretende reconocer los números del 0 al 9, alimentada con la información de números escritos a mano. Por lo tanto, la capa de salida será un array de 1x10, es decir, los diez resultados esperados.

En cuanto a la capa de inicio, un array de tamaño 784 (28*28) es el que recibe la información procesada de una imagen de 28x28 píxeles que contiene el número escrito a mano.
Básicamente, podemos imaginar que dividimos la imagen en 784 celdas y se le da un valor 0 donde no hay tinta, y otro valor donde sí la hay.

Lo que ocurre entre esta primera capa de inicio y la capa de salida con el resultado esperado es un asunto de probabilidades.

Inicié esta red con una capa de inicio de 784 y una sola capa intermedia de 64 neuronas:
(784) -> (64) -> (10)
Y una función sigmoide para ajustar los valores obtenidos de las entradas, sus pesos y umbrales, en valores entre 0 y 1 (sin incluir el 0 y 1), y llevarlos a la matriz que resultaba en la capa siguiente.
Y aunque resultó, la curva de aprendizaje solo fue óptima a valores altos de learning_rate, lo que no me hizo esperar a obtener problemas de estancamiento para comprender que necesitaba una nueva capa neuronal.

Así fue que agregué, reajusté la primera capa intermedia a 128 y agregué una nueva capa de 64:
(784) -> (128) -> (64) -> (10)
Lo que me dio problemas con la función sigmoide. Básicamente, el gradiente aplicado a los valores de la capa de salida era mínimo, lo que estancaba la curva de aprendizaje.

<br/>
Epoch 0, Loss: 2.3037<br/>
Epoch 10, Loss: 2.3012<br/>
Epoch 20, Loss: 2.3012<br/>
Epoch 30, Loss: 2.3012<br/>
Epoch 40, Loss: 2.3012<br/>
Epoch 50, Loss: 2.3012<br/>
Epoch 60, Loss: 2.3012<br/>
Epoch 70, Loss: 2.3012<br/>
Epoch 80, Loss: 2.3012<br/>
Epoch 90, Loss: 2.3012<br/>
Loss: 2.3011508437188146 Una mierda<br/>
<br/>
ChatGPT me sugirió cambiar a ReLU. Y aunque sabía que la función sigmoide estaba obsoleta en este campo, la había elegido para comenzar a aprender redes neuronales. En resumen, no valió la pena. Pero es bueno comprender lo que es una función de gradientes aplicada a este campo, para saber en qué dirección debe moverse la red para obtener el resultado correcto.

Llegado a este punto, conviene aclarar que los resultados de la red no son una matriz con el valor 1 en el resultado esperado y 0 en el resto.
El resultado es un array de 10 elementos, con un valor entre 0 y 1 que representa la probabilidad de que cada elemento sea el resultado correcto. Imaginemos que se le da a la red la entrada de datos correspondiente al número 3. El resultado esperado de la capa podría asimilarse a esto:
    output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
Y al ser el valor más alto el del índice 3, 0.85, la red asumirá que el número es el 3.

Entre la capa de entrada (con 784 neuronas) y la capa oculta siguiente (128 neuronas) existen 100.352 conexiones posibles.
La relación entre estas capas puede calcularse y pensarse como un nuevo array bidimensional o matriz.
Dicha relación se establece de la siguiente manera:
Neurona_inicio_x * Wi + Bi, donde W es el peso que tiene dicha relación y B un sesgo de activación para la neurona resultante.

En nuestro set de datos de entrenamiento, que viene en formato CSV, tenemos una primera columna "label" que nos indica qué número se representará con las siguientes columnas "pixel_x" (pixel_1, pixel_2, pixel_3...).
De manera que donde haya un trazo de escritura, se representará con un número, y donde no, con un cero.
Estos datos se preparan separando la columna en un nuevo arreglo y_train:
    y_train = train['label'].values
Y en una matriz con los valores de los píxeles, sin esta columna, x_train:
    x_train = train.drop(columns=['label']).values / 255.0
Como el valor máximo es de 255, se normalizan los datos dividiéndolos por 255 para obtener como número máximo un 1 y el resto flotantes mayores a 0. Es decir, x_train contiene números entre 0 y 1.

Luego se inicia una matriz de tamaño 784x128 (100.352) inicializada con números al azar:
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
Esta es la matriz de pesos asociados a las conexiones. Se inicializan de manera aleatoria, pero con la etapa de entrenamiento esos valores se irán ajustando.
También se inicializa un array de tamaño 128 (128 para esta primera capa) completado con ceros. Este es el array b1 que contiene los sesgos de activación para cada neurona:
    b1 = np.zeros((1, hidden_size))
De igual manera, se ajustan durante el entrenamiento.

Por lo tanto, de los valores obtenidos de pixel_x * Wi + bi se forma la nueva matriz representante de la primera capa oculta:
    Z1=np.dot(x_train, W1)+b1

Aquí, luego, procesaba esta matriz con la función sigmoide para comprimir los valores a una escala entre 0 y 1. Pero, como dije antes, eso se cambió a la función ReLU.

La función sigmoide, como dije antes, comprimía los valores en una escala entre 0 y 1. En cambio, la función ReLU retorna 0 cuando un valor es menor a 0, y retorna x cuando el valor x es mayor a 0.

La matriz obtenida entonces, con valores superiores a 0, resulta en la capa neuronal A1:
    Z1=np.dot(x_train, W1)+b1
    A1=relu(Z1)
El proceso entonces se repite la cantidad de veces correspondiente a la cantidad de capas neuronales.

Finalmente, obtengo la capa final, no aplicando la función ReLU, que es solo para las capas ocultas, sino la función Softmax.

Softmax calcula la matriz de probabilidades. En esta matriz, el número de valor más alto es el resultado que se estima correcto.

Softmax: ![alt text](/miscellaneous/image.png)

Softmax tomará la matriz resultante de la capa neuronal oculta previa a la salida, y tomará cada elemento de esa matriz, usándolo como exponente para elevar e a ese exponente y lo dividirá por la sumatoria de todos e elevado a cada elemento de esa misma matriz. (arriba en la imagen la formula matematica **).

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

Imaginemos que Z3 es solo un array de 2 elementos: Z3 = [Z1, Z2] = [1.0, 2.0]

Calculamos e^Z1 y e^Z2: e^1, e^2 = 2.718, 7.389.

El resultado será otra matriz A3 de 2 elementos cuyos valores son:

A3 = [A1, A2]
A1 = 2.718/(2.718 + 7.389) = 0.2689, A2 = 7.389/(2.718 + 7.389) = 0.7311
A3 = [0.2689, 0.7311]

Calculando la pérdida

Una vez que tenemos el array de probabilidades de acierto (A3), queremos saber, si no acertó, qué ajustes debe hacer a los pesos asociados a la respuesta. Para eso se utiliza cálculo de gradientes, al menos para este tipo de red.

En realidad, si acierta también realiza ajustes, o si no acierta, realiza ajustes en todos los pesos, no solo los asociados a los valores esperados. Pero estos son tan insignificantes que realmente podemos desestimarlos en la explicación.

Supongamos que en A3 tiene estos valores:
[[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
Y que el label correspondiente a los datos de entrada que produjo A3 fue [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
En A3 se interpreta entonces que el resultado correcto es 3, pero en realidad sabemos por el label que es 2.

Teniendo la matriz de probabilidades A3, se procede a multiplicarla por la matriz de los labels y_train_encoded:
A3*y_train_encoded = [0x0.03, 0x0.01, 1x0.02, 0x0.85, 0x0.02, 0x0.01, 0x0.02, 0x0.02, 0x0.04, 0x0.02] = [0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0]
En realidad, se multiplica por el logaritmo de la predicción correcta. Pero no voy a calcular el logaritmo de cada predicción para el ejemplo. Sin embargo, en la función se observa el método:

    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8))/m

Se suman y se divide por la cantidad de elementos del batch, m, para obtener el promedio. Con ese valor se ajustan los pesos asociados a la respuesta que debe dar.

Finalmente obtuve:
<br/>
Epoch 0, Loss: 3.2855<br/>
Epoch 10, Loss: 2.0808<br/>
Epoch 20, Loss: 1.6778<br/>
Epoch 30, Loss: 1.2275<br/>
Epoch 40, Loss: 1.2251<br/>
Epoch 50, Loss: 0.9796<br/>
Epoch 60, Loss: 0.7584<br/>
Epoch 70, Loss: 0.6778<br/>
Epoch 80, Loss: 0.6045<br/>
Epoch 90, Loss: 0.5534<br/>
Loss:  0.5166312820443583<br/>
<br/>

Para finalizar, luego de la etapa de entrenamiento sigue la validación y la inferencia, pero no es necesario profundizar en ellas, ya que son lo mismo, pero sin el ajuste que conlleva la etapa de aprendizaje.

* No sé qué variación tiene en Python un array bidimensional, una matriz o un mapa respecto al tipo de datos. Si hablo de uno u otro, quiero que se entienda que no hago diferenciación, ya que lo dejo en plano algebraico. No quiere decir que suponga que son el mismo tipo de datos.

** En el código, la sumatoria se realiza elevando e a (y_pred + 1e-8) para evitar el overflow numérico.
