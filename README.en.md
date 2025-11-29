First of all, it’s worth clarifying that rather than technical specifications, I will leave here some considerations more closely related to the basic notions of how this works.
As a way to revisit what I’ve learned, here I explain it to myself and to anyone who visits this repository.

It is often joked that AI is a nest of “IF” loops.  
Within AI, we could say—without so much joking—that the neural network designed in this repository is a process of applied statistics over matrices (or, in Python terms, 2D arrays*) in a sequential manner, thus obtaining a final layer matrix where the highest value is the expected result.

Basically, what is coded here is a group of matrices that interact with each other, but during training they are given the correct answers so that those relationships can later be adjusted.

This neural network aims to recognize numbers from 0 to 9, fed with data consisting of handwritten digits. Therefore, the output layer will be a 1x10 array, that is, the ten expected results.

Regarding the input layer, an array of size 784 (28*28) receives the processed information from a 28x28-pixel image containing the handwritten digit.  
Basically, we can imagine dividing the image into 784 cells and assigning a value of 0 where there is no ink, and another value where there is.

What happens between this first input layer and the output layer with the expected result is a matter of probabilities.

I began this network with an input layer of 784 and a single hidden layer of 64 neurons:
    (784) -> (64) -> (10)
And a sigmoid function to adjust the values obtained from the inputs, their weights, and thresholds into values between 0 and 1 (not including 0 and 1), and pass them to the matrix forming the next layer.
And although it worked, the learning curve was only optimal at high learning_rate values, which made it unnecessary to wait for stagnation before realizing that I needed another neural layer.

So I added one: I readjusted the first hidden layer to 128 and added a new layer of 64:
    (784) -> (128) -> (64) -> (10)
This introduced problems with the sigmoid function. Basically, the gradient applied to the output layer values was minimal, which stalled the learning curve.

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
Loss: 2.3011508437188146 — Crap<br/>
<br/>

ChatGPT suggested switching to ReLU. And although I knew that the sigmoid function is obsolete in this field, I chose it when starting to learn neural networks. In short, it wasn’t worth it. But it is good to understand what a gradient function is in this field, to see in which direction the network must move to obtain the correct result.

At this point, it’s worth clarifying that the network outputs are not a matrix with a value of 1 for the correct result and 0 for the rest.  
The result is an array of 10 elements with a value between 0 and 1 that represents the probability of each element being the correct answer.  
Imagine we give the network the input corresponding to number 3. The expected output of the layer could look like this:

    output_probs = [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
    
Since the highest value, 0.85, is at index 3, the network will assume the number is 3.

Between the input layer (with 784 neurons) and the next hidden layer (128 neurons) there are 100,352 possible connections.  
The relationship between these layers can be computed and thought of as a new 2D array or matrix.  
This relationship is established as follows:

    Input_neuron_x * Wi + Bi,
    
where W is the weight of that connection and B is an activation bias for the resulting neuron.

In our training dataset, which comes in CSV format, we have a first column “label” that indicates which digit is represented by the following columns “pixel_x” (pixel_1, pixel_2, pixel_3…).  
Where there is handwriting, we get a number; where not, a zero.  
This data is prepared by separating the column into a new array y_train:

    y_train = train['label'].values
    
And into a matrix containing only pixel values, without this column, x_train:

    x_train = train.drop(columns=['label']).values / 255.0
    
Since the maximum value is 255, we normalize the data by dividing by 255 so the maximum becomes 1, and the rest are floats greater than 0. In other words, x_train contains numbers between 0 and 1.

Then a matrix of size 784x128 (100,352 values) is initialized with random numbers:

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    
This is the weight matrix associated with the connections. They are initialized randomly, but during training these values will be adjusted.  
An array of size 128 (for this first hidden layer) filled with zeros is also initialized. This is array b1 and contains the activation biases for each neuron:

    b1 = np.zeros((1, hidden_size))
    
Likewise, these are adjusted during training.

Thus, from the values obtained using pixel_x * Wi + bi, the new matrix representing the first hidden layer is formed:

    Z1 = np.dot(x_train, W1) + b1

Here, I then processed this matrix with the sigmoid function to compress values into a 0–1 range. But, as mentioned earlier, this was replaced with the ReLU function.

The sigmoid function, as I said, compresses values into the 0–1 range. In contrast, ReLU returns 0 when a value is less than 0, and returns x when x is greater than 0.

The resulting matrix, containing values above 0, becomes the neural layer A1:

    Z1 = np.dot(x_train, W1) + b1
    A1 = relu(Z1)
    
This process repeats as many times as there are neural layers.

Finally, I obtain the final layer, not using ReLU (which is only for hidden layers), but the Softmax function.

Softmax computes the probability matrix. In this matrix, the highest value is the result estimated as correct.

Softmax: ![alt text](/miscellaneous/image.png)

Softmax will take the matrix resulting from the previous hidden layer and take each element of this matrix, using it as an exponent to raise e to that value, dividing it by the sum of all e raised to every element in that same matrix. (The mathematical formula is shown above in the image**).

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

Imagine Z3 is just an array of 2 elements: Z3 = [Z1, Z2] = [1.0, 2.0]

We compute e^Z1 and e^Z2: e^1, e^2 = 2.718, 7.389.

The result will be another matrix A3 of two elements whose values are:

    A3 = [A1, A2]
    A1 = 2.718/(2.718 + 7.389) = 0.2689, A2 = 7.389/(2.718 + 7.389) = 0.7311
    A3 = [0.2689, 0.7311]

Calculating the loss

Once we have the array of probability outputs (A3), we want to know—if it didn’t guess correctly—what adjustments must be made to the weights associated with the answer. For this, gradient calculation is used, at least for this type of network.

In reality, even when it guesses correctly it still performs adjustments, and if it doesn’t, it still adjusts all weights, not just those associated with the expected values.  
But these are so insignificant that we can safely ignore them for explanation.

Suppose A3 has these values:

    [[0.03, 0.01, 0.02, 0.85, 0.02, 0.01, 0.02, 0.02, 0.04, 0.02]]
    
And the label corresponding to the input that produced A3 was [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].  
A3 interprets that the correct result is 3, but we know from the label that it is 2.

Given the probability matrix A3, we proceed to multiply it by the label matrix y_train_encoded:

    A3*y_train_encoded = [0x0.03, 0x0.01, 1x0.02, 0x0.85, 0x0.02, 0x0.01, 0x0.02, 0x0.02, 0x0.04, 0x0.02] = [0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0]
    
In reality, we multiply by the logarithm of the correct prediction. But I’m not computing logarithms here for the example. However, the actual function shows the method:

    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8))/m

Values are summed and divided by the number of elements in the batch, m, to obtain the average. With that value, the weights associated with the output are adjusted.

Finally I obtained:
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

To conclude, after the training stage comes validation and inference, but there is no need to go deep into them, as they are the same process but without the adjustments associated with the learning stage.

* I do not know what variation exists in Python between a 2D array, a matrix, or a map in terms of data types. If I refer to one or the other, I want it to be understood that I am not making a distinction, as I treat them in an algebraic sense. This does not mean I assume they are the same data type.

** In the code, the summation is performed by raising e to (y_pred + 1e-8) to avoid numerical overflow.
