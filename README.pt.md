Primeiramente, é importante esclarecer que, mais do que especificações técnicas, deixarei aqui algumas considerações mais próximas das noções básicas de seu funcionamento.
Como forma de recordar o que foi aprendido, passo aqui a explicar para mim mesmo e para quem visitar este repositório.

Já foi dito, em tom de brincadeira, que a IA é um ninho de laços “IF”.
Dentro das IAs, pode-se dizer — sem tanta brincadeira — que a rede neural desenvolvida neste repositório é um processo de estatística aplicada a matrizes (ou, para Python, arrays bidimensionais*) de maneira sequencial, obtendo assim uma última matriz de camada onde o valor mais alto é o resultado esperado.

Basicamente, o que se codifica aqui é um conjunto de matrizes que se inter-relacionam, mas que, no momento do treinamento, recebem as respostas corretas para que essas relações possam ser ajustadas posteriormente.

Esta rede neural tem como objetivo reconhecer os números de 0 a 9, alimentada com informações de dígitos escritos à mão. Portanto, a camada de saída será um array de 1x10, ou seja, os dez resultados possíveis.

Quanto à camada de entrada, trata-se de um array de tamanho 784 (28*28), que recebe a informação processada de uma imagem de 28x28 pixels contendo o número escrito à mão.
Basicamente, podemos imaginar que dividimos a imagem em 784 células e damos valor 0 onde não há tinta e outro valor onde há.

O que ocorre entre essa primeira camada de entrada e a camada de saída com o resultado esperado é uma questão de probabilidades.

Iniciei esta rede com uma camada de entrada de 784 e apenas uma camada intermediária de 64 neurônios:
    (784) -> (64) -> (10)
E uma função sigmóide para ajustar os valores obtidos das entradas, seus pesos e biases em valores entre 0 e 1 (sem incluir 0 e 1), levando-os à matriz resultante da camada seguinte.
E embora tenha funcionado, a curva de aprendizado só foi adequada com valores altos de learning_rate, o que me fez prever que teríamos problemas de estagnação — motivo pelo qual concluí que eu precisava de uma nova camada neural.

Assim, reajustei a primeira camada intermediária para 128 e adicionei uma nova camada de 64:
    (784) -> (128) -> (64) -> (10)
Isso, porém, trouxe problemas com a função sigmóide. Basicamente, o gradiente aplicado aos valores da camada de saída era mínimo, o que estagnava a curva de aprendizado.

<br/> Epoch 0, Loss: 2.3037<br/> Epoch 10, Loss: 2.3012<br/> Epoch 20, Loss: 2.3012<br/> Epoch 30, Loss: 2.3012<br/> Epoch 40, Loss: 2.3012<br/> Epoch 50, Loss: 2.3012<br/> Epoch 60, Loss: 2.3012<br/> Epoch 70, Loss: 2.3012<br/> Epoch 80, Loss: 2.3012<br/> Epoch 90, Loss: 2.3012<br/> Loss: 2.3011508437188146 Uma merda<br/> <br/>

O ChatGPT sugeriu trocar para ReLU. E embora eu soubesse que a função sigmóide está obsoleta neste campo, eu a havia escolhido para aprender redes neurais desde o básico. Resumo: não valeu a pena. Mas é útil entender o que é uma função de gradientes aplicada a este campo, para saber em que direção a rede deve se mover para obter o resultado correto.

Chegado a este ponto, convém esclarecer que os resultados da rede não são uma matriz com valor 1 no resultado esperado e 0 nos demais.
O resultado é um array de 10 elementos, com um valor entre 0 e 1 que representa a probabilidade de que cada elemento seja o correto.
Imaginemos que fornecemos à rede um conjunto de dados correspondente ao número 3. O resultado esperado da camada poderia ser algo assim:

    output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]

E, sendo o valor mais alto o do índice 3, 0.85, a rede assumirá que o número é 3.

Entre a camada de entrada (784 neurônios) e a seguinte camada oculta (128 neurônios) existem 100.352 conexões possíveis.
A relação entre essas camadas pode ser pensada e calculada como uma nova matriz bidimensional.
Essa relação é estabelecida da seguinte forma:

    Neurônio_inicial_x * Wi + Bi,

onde W é o peso dessa relação e B é um bias de ativação para o neurônio resultante.

No nosso dataset de treinamento, que vem em formato CSV, temos uma primeira coluna label indicando qual número será representado pelas colunas seguintes pixel_x (pixel_1, pixel_2, pixel_3...).
Ou seja, onde houver traço de escrita, haverá um valor numérico; onde não houver, um zero.
Esses dados são preparados separando a coluna em um novo array y_train:

    y_train = train['label'].values

E em uma matriz com os valores dos pixels, sem essa coluna, x_train:

    x_train = train.drop(columns=['label']).values / 255.0

Como o valor máximo é 255, os dados são normalizados dividindo por 255 para obter valores entre 0 e 1.

Em seguida, inicia-se uma matriz de tamanho 784x128 (100.352) inicializada com números aleatórios:

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)

Essa é a matriz de pesos das conexões. Eles começam como valores aleatórios, mas se ajustam durante o treinamento.
Também se inicializa um array de tamanho 128 preenchido com zeros, o array b1 que contém os biases:

    b1 = np.zeros((1, hidden_size))

Da mesma forma, serão ajustados durante o treinamento.

Portanto, dos valores obtidos de pixel_x * Wi + bi se forma a nova matriz que representa a primeira camada oculta:

    Z1 = np.dot(x_train, W1) + b1

Aqui, eu aplicava a função sigmóide para comprimir os valores entre 0 e 1. Mas, como mencionei antes, isso foi substituído por ReLU.

A função sigmóide comprimia os valores para uma escala entre 0 e 1. Já a função ReLU retorna 0 quando o valor é menor que 0 e retorna x quando o valor é maior que 0.

A matriz resultante, com valores maiores que 0, transforma-se então na camada neuronal A1:

    Z1 = np.dot(x_train, W1) + b1
    A1 = relu(Z1)

O processo se repete tantas vezes quanto o número de camadas neuronais.

Finalmente, obtenho a camada final, não aplicando a função ReLU — que é usada apenas nas camadas ocultas — mas sim a função Softmax.

Softmax calcula a matriz de probabilidades. Nessa matriz, o número de valor mais alto é o resultado presumido como correto.

Softmax: ![alt text](/miscellaneous/image.png)

Softmax pega a matriz resultante da camada oculta anterior à saída, eleva cada elemento como expoente de e, e divide pelo somatório de todos esses expoentes (na imagem acima, está a fórmula matemática**).

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

Imaginemos que Z3 é apenas um array de 2 elementos: Z3 = [Z1, Z2] = [1.0, 2.0]

Calculamos e^Z1 e e^Z2: e^1, e^2 = 2.718, 7.389.

O resultado será outro array A3 de 2 elementos:

    A3 = [A1, A2]
    A1 = 2.718/(2.718 + 7.389) = 0.2689
    A2 = 7.389/(2.718 + 7.389) = 0.7311
    A3 = [0.2689, 0.7311]

Calculando a perda

Uma vez que temos o array de probabilidades (A3), queremos saber, se errou, quais ajustes devem ser feitos nos pesos. Para isso se utiliza cálculo de gradientes.

Na verdade, mesmo acertando, a rede também ajusta os pesos, só que esses ajustes são tão pequenos nos valores não associados ao resultado esperado que podemos ignorá-los aqui.

Suponhamos que A3 tenha estes valores:

    [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]

E que o label correspondente aos dados de entrada seja [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
A rede interpreta que o resultado correto é 3, mas sabemos pelo label que é 2.

Tendo a matriz A3, multiplicamos pelo array y_train_encoded:

    A3 * y_train_encoded = [0x0.03, 0x0.01, 1x0.02, 0x0.85, ...] = [0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0]

Na verdade, multiplica-se pelo logaritmo da predição correta. Não vou calcular os logs aqui, mas no código fica assim:

    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

Somam-se os valores, divide-se pelo tamanho do batch m e o resultado é usado para ajustar os pesos da rede.

Finalmente, obtive:
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
Loss: 0.5166312820443583<br/>
<br/>

Para finalizar, depois da etapa de treinamento vem a validação e a inferência, mas não é necessário aprofundar aqui, já que são os mesmos processos, apenas sem os ajustes da etapa de aprendizado.

*Não sei qual variação existe em Python entre array bidimensional, matriz ou mapa. Se eu usar um termo ou outro, quero deixar claro que não faço distinção, pois estou tratando no âmbito algébrico. Não significa que eu suponha que sejam o mesmo tipo de dado.

**No código, a soma é feita elevando e a (y_pred + 1e-8) para evitar overflow numérico.
