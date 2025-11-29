Primeiramente, vale esclarecer que, mais do que especificações técnicas, deixarei aqui algumas considerações voltadas às noções básicas do funcionamento da rede. Como forma de registrar o que aprendi, passo a explicar para mim mesmo e para quem visitar este repositório.

Já foi dito em tom de brincadeira que IA é um ninho de loops “IF”. Dentro do contexto de IA, pode-se dizer — sem tanta brincadeira — que a rede neural construída neste repositório é um processo de estatística aplicada a matrizes (ou, em Python, arrays bidimensionais*) de maneira sequencial, obtendo assim uma matriz final de saída, onde o valor mais alto representa o resultado esperado.

Basicamente, o que é codificado aqui é um conjunto de matrizes que se relacionam entre si, mas que, durante o treinamento, recebem as respostas corretas para que essas relações sejam ajustadas.

Esta rede neural busca reconhecer os números de 0 a 9, usando como entrada informações de dígitos escritos à mão. Portanto, a camada de saída será um array de 1x10, isto é, os dez resultados possíveis.

Quanto à camada de entrada, é um array de tamanho 784 (28*28), que recebe a informação processada de uma imagem 28x28 pixels contendo o número escrito à mão. Podemos imaginar que dividimos a imagem em 784 células, atribuindo valor 0 onde não há tinta e outro valor onde há.

O que acontece entre essa primeira camada e a camada final é uma questão de probabilidades.

Comecei esta rede com uma camada de entrada de 784 e apenas uma camada intermediária de 64 neurônios:
    (784) -> (64) -> (10).
E usei uma função sigmoide para ajustar os valores das entradas, seus pesos e seus vieses para uma escala entre 0 e 1 (sem incluir exatamente 0 e 1), alimentando a matriz da próxima camada. Funcionou, mas a curva de aprendizado só ficou aceitável com valores altos de learning_rate — o que já indicava, antes mesmo de problemas de estagnação, que eu precisava de outra camada.

Então reajustei a primeira camada intermediária para 128 neurônios e adicionei outra camada de 64:
    (784) -> (128) -> (64) -> (10).
Isso gerou problemas com a função sigmoide. Basicamente, o gradiente aplicado aos valores da saída era mínimo, o que travava a curva de aprendizado.

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
  Loss: 2.3011508437188146  Uma merda.<br/>
  <br/>
O ChatGPT sugeriu mudar para ReLU. E embora eu já soubesse que a sigmoide está ultrapassada nesse campo, a escolhi para começar a aprender redes neurais do zero. Em resumo: não valeu a pena. Mas é bom entender o que é uma função de gradiente neste contexto, para saber em que direção a rede deve se ajustar para chegar ao resultado certo.

Aproveito para esclarecer: a saída da rede não é uma matriz com valor 1 para o resultado correto e 0 para os outros. O resultado é um array de 10 elementos, cada um com um valor entre 0 e 1 que representa a probabilidade de cada classe ser a correta. Suponha que fornecemos à rede o número 3. A saída esperada poderia ser algo como:

    output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]

Como o maior valor é o índice 3 (0.85), a rede assume que o número é 3.

Entre a camada de entrada (784 neurônios) e a camada oculta seguinte (128 neurônios) existem 100.352 conexões possíveis. Essa relação pode ser pensada como uma matriz. Cada neurônio da próxima camada é calculado como:

    Neurônio_inicial_x * Wi + Bi

Onde W é o peso da conexão e B é o viés da neurona.

No dataset de treinamento (CSV), a primeira coluna “label” indica qual número será representado pelas próximas colunas “pixel_x” (pixel_1, pixel_2, pixel_3...). Onde há traço, há um valor; onde não há, 0. Preparamos esses dados assim:

    y_train = train['label'].values
    x_train = train.drop(columns=['label']).values / 255.0

O valor máximo é 255, então normalizamos para obter valores entre 0 e 1.

Depois iniciamos uma matriz 784x128 com valores aleatórios:

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)

E um array de 128 zeros (viés):

    b1 = np.zeros((1, hidden_size))

A primeira camada oculta é:

    Z1 = np.dot(x_train, W1) + b1

Antes eu processava essa matriz com a função sigmoide, mas isso foi trocado por ReLU.

A sigmoide comprimia para (0, 1). Já a ReLU retorna 0 quando x < 0 e retorna x quando x > 0.

Assim, a camada A1 fica:

    A1 = relu(Z1)

O processo se repete para as demais camadas ocultas.

A camada final usa Softmax, não ReLU.

Softmax calcula probabilidades. O maior valor é o resultado final.

Exemplo simples:

    Z3 = [1.0, 2.0]
    e^1 = 2.718
    e^2 = 7.389
    A3 = [0.2689, 0.7311]

Cálculo da perda

Uma vez obtido o array A3, queremos saber que ajustes devem ser feitos nos pesos quando erramos (ou até mesmo quando acertamos). Para isso usamos gradientes e a função cross-entropy:

    def cross_entropy(y_true, y_pred):
      m = y_true.shape[0]
      return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

Exemplo simplificado:

    A3 = [0.03, 0.01, 0.02, 0.85, ...]
  
    Label one-hot para “2”: [0, 0, 1, 0, ...]

Multiplicamos elemento a elemento:

    A3 * y_true = [0, 0, 0.02, 0, ...]

E aplicamos o logaritmo apenas do valor correspondente ao índice correto.

Ao final do treinamento, obtive:

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

Para finalizar: depois da etapa de treinamento vem a validação e a inferência, mas não é necessário aprofundar, pois o processo é o mesmo — só não há ajuste dos pesos.

Não sei exatamente qual variação Python faz entre array bidimensional, matriz ou mapa enquanto tipos de dados. Se falo de um ou outro, quero deixar claro que não estou diferenciando no sentido algébrico — não significa que suponho serem o mesmo tipo na prática.

No código, a soma usa e^(y_pred + 1e-8) para evitar overflow numérico.
