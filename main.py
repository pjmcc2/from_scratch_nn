import autograd.numpy as np
from numpy.random import default_rng
from sklearn import datasets as ds
from autograd import elementwise_grad




def one_hot_encode(labels):
    """
    One hot encodes labels.
    :param labels:
    The labels to be encoded. List or numpy array etc.
    :return:
    The encoded labels as list of sparce arrays
    """
    new_labels = []
    num_labels = len(np.unique(labels))
    for label in labels:
        zero_array = np.zeros(num_labels)
        zero_array[label] = 1
        new_labels.append(zero_array)

    return new_labels


def randomize_input(features, labels, random):
    """
    Paired psuedo-randomization of features and labels

    :param features:
    Data features to be randomized. Subscriptable
    :param labels:
    Data labels to be randomized. Subscriptable
    :param random:
    The numpy RNGenerator. Must be given.
    :return:
    Shuffled Data and labels using numpy's indexing
    """
    rng = random
    indices = np.arange(len(features))
    rng.shuffle(indices)

    return features[indices], labels[indices]


def v_relu(v):  # activation function
    """
    Rectified Linear Unit Activation Function vectorized by numpy
    Works with batches.
    :param v:
    Vector to be activated.
    :return:
    Activated vector.
    """
    new_array = []
    if len(v.shape) <= 2:
        for i in range(len(v)):
            t = [0.0] if v[i] <= 0 else v[i]
            new_array.append(t)
        return np.array(new_array, ndmin=1)
    else:
        assert len(v.shape) == 3  # only 3 dimensions please
        for i in range(v.shape[0]):
            temp = []
            for c in v[i].flatten():
                t = [0.0] if c <= 0 else [c]
                temp.append(t)
            new_array.append(np.array(temp))
    return np.array(new_array)


def accuracy(layers, iv, label):
    """
    Accuracy metric.
    :param layers:
    List of weights, bias
    :param iv:
    Input vector(s)
    :param label:
    Label(s) corresponding to input vector(s).
    :return:
    Mean loss and mean accuracy (if using minibatch size >1)
    Loss and accuracy if minibatch size == 1.
    """
    guess = softmax(forward_pass(layers, iv))
    loss = cross_entropy(layers, guess, label, grad=False)
    return np.mean(loss), np.mean(np.argmax(label, axis=1) == np.argmax(guess, axis=1))


def softmax(x):
    """
    Softmax activation function.
    :param x:
    Input vector(s) t0 be activated.
    :return:
    activated input vector(s)
    """
    def sub_soft(v):
        z = v - np.max(v)
        num = np.exp(z)
        denom = np.sum(num)
        return num / denom

    if len(x.shape) <= 2:
        return sub_soft(x)
    else:
        assert len(x.shape) == 3
        temp = []
        for v in x:
            t = sub_soft(v)
            temp.append(t)
        return np.array(temp)


def cross_entropy(layers, input, labels, grad=False):  # from softmax #TODO REORDER ARGS
    """
    Cross-Entropy loss function. Used for directly calculating loss, and for calculating the gradient with
    autograd.
    :param layers:
     List of weights and biases, only used for calculating gradient with autograd.
    :param input:
    input vector(s). Also only used for calculating gradient w/ autograd.
    :param labels:
    Ground truth labels.
    :param grad:
    Boolean to determine if used for autgrad
    :return:
    Cross Entropy Loss as np array
    """
    if grad:
        probs = forward_pass(layers, input)
    else:
        probs = input

    temp = []
    for i in range(len(labels)):
        loss = -np.sum(labels[i] * np.log(probs[i]))
        temp.append(loss)

    return np.array(temp)


def update_weights_and_biases(params, grad, step):
    """
    Update weights and biases by stepsize.
    :param params:
    List of weights and biases
    :param grad:
    Gradient. List of numpy arrays.
    :param step:
    Step size.
    :return:
    None.
    """
    for i in range(len(params)):
        w, b = params[i]
        w_grad, b_grad = grad[i]
        w_grad = np.mean(w_grad, axis=0)
        b_grad = np.mean(b_grad, axis=0)
        w -= step * w_grad
        b -= step * b_grad


def bmm(a, b):
    """
    Batch Matrix Multiply
    :param a:
    Left Matrix
    :param b:
    Right Matrix
    :return:
    Batch multiplied AB
    """
    return np.einsum("ijk, ikl->ijl", a, b)


def forward_pass(params, input_v):
    """
    Pass through the network.
    :param params:
    List of weights and Biases.
    :param input_v:
    Input Vector
    :return:
    Softmax "probabilities" as np array.
    """
    curr_vector = input_v
    for w, b in params:
        if len(w.shape) > 2:
            weighted_sum = bmm(w, curr_vector) + b
        else:
            weighted_sum = w @ curr_vector + b
        curr_vector = v_relu(weighted_sum)

    probs = softmax(weighted_sum)
    return probs


def batch_input(data, batch_size):
    """
    Organizes data into batches.
    :param data:
    List of Data to be organized. (features and labels)
    :param batch_size:
    Size of batches.
    :return:
    list of np arrays.
    """
    temp = []
    temp2 = []
    output = []
    for dataset in data:
        count = 0
        for el in dataset:
            count += 1
            temp.append(el)
            if count == batch_size:
                temp2.append(np.array(temp))
                temp = []
                count = 0
        if len(temp) != 0: # check for stragglers
            for i in range(batch_size - len(temp)):
                temp.append(dataset[i])  # "wrap around" and repeat first few examples to fill out last batch
            temp2.append(np.array(temp))
        output.append(np.array(temp2))
        temp2 = []

    return output


def initialize_layers(num_hlayers, num_weights, input_size, output_size, batch_size):
    """
    Function to control the creation of layers.
    :param num_hlayers:
    How many hidden layers to be included in the network. Total number of layers is 2 + num_hlayers
    :param num_weights:
    Number of weights per layer (excluding input layer)
    :param input_size:
    Size (num elements) of input vector
    :param output_size:
    Size (num elements) of output vector. MUST correspond to num of classes
    :param batch_size:
    Size of mini-batches. Each batch is used to compute average gradient.
    :return:
    List of Weights and Biases as np arrays
    """
    # Input layer with special size requirements: num_weights x input_size
    layers = []
    w = rng.normal(0, 2 / input_size, (num_weights, input_size))
    b = rng.standard_normal((num_weights, 1))
    bw = np.tile(w, (batch_size, 1, 1))
    bb = np.tile(b, (batch_size, 1, 1))
    layers.append((bw, bb))
    # Hidden layers all share same size
    for i in range(num_hlayers - 2):
        w = rng.normal(0, 2 / num_weights, (num_weights, num_weights))
        b = rng.standard_normal((num_weights, 1))
        bw = np.tile(w, (batch_size, 1, 1))
        bb = np.tile(b, (batch_size, 1, 1))
        layers.append((bw, bb))
    # Output layer has special size requirement: output_size x num_weights
    w = rng.normal(0, 2 / num_weights, (output_size, num_weights))
    b = rng.standard_normal((output_size, 1))
    bw = np.tile(w, (batch_size, 1, 1))
    bb = np.tile(b, (batch_size, 1, 1))
    layers.append((bw, bb))

    return layers


def train_loop(gradient_func, layers, input_vectors, labels, epochs, step_size):
    """
    Loop that trains the network!
    :param gradient_func: autograd gradient computation function
    :param layers:
    list of layers.
    :param input_vectors:
    Batched input
    :param labels:
    Batched Labels
    :param epochs:
    Num of iterations across entire training set.
    :param step_size:
    step size of updates to weights and biases
    :return:
    None.
    """
    for j in range(epochs):
        epoch_loss = 0.0
        epoch_accr = 0.0
        for i in range(len(input_vectors)):
            gradient = gradient_func(layers, input_vectors[i], labels[i], grad=True)
            update_weights_and_biases(layers, gradient, step_size)
            train_loss, train_acc = accuracy(layers, input_vectors[i], labels[i])
            epoch_loss += (train_loss - epoch_loss) / (i + 1)
            epoch_accr += (train_acc - epoch_accr) / (i + 1)
        print("Epoch {epoch}, train loss: {l}  train acc: {a}".format(epoch=j, l=epoch_loss, a=epoch_accr))


def unbatch_layers(layers):
    """
    If you want to unbatch your layers!
    :param layers:
    batched layers
    :return:
    Unbatched layers, e.g. weights with shape (batch_size, N, M) to (N, M)
    """
    new_layers = []
    for i in range(len(layers)):
        w, b = layers[i]
        new_layers.append((w[0], b[0]))

    return new_layers


def test_loop(layers, input, labels, test_batches=False):
    """
    Loop to test network on validation set.
    :param layers:
    trained layers as list of tuples of np arrays
    :param input:
    input vectors
    :param labels:
    corresponding labels
    :param test_batches:
    Boolean to see if you would like to test in batches or not.
    :return:
    None
    """
    m_loss = 0.0
    m_acc = 0.0
    if test_batches:
        for i in range(len(input)):
            test_loss, test_acc = accuracy(layers, input[i], labels[i])
            m_loss += (test_loss - m_loss) / (i + 1)
            m_acc += (test_acc - m_acc) / (i + 1)
    else:
        new_l = unbatch_layers(layers)
        for i in range(len(input)):
            test_loss, test_acc = accuracy(new_l, input[i], labels[i])
            m_loss += (test_loss - m_loss) / (i + 1)
            m_acc += (test_acc - m_acc) / (i + 1)
    print("Validation loss: {l}, val acc: {a}".format(l=m_loss, a=m_acc))


if __name__ == "__main__":
    # Create numpy random number generator
    rng = default_rng(1337)

    # Load data
    iris_X, iris_y = ds.load_iris(return_X_y=True)
    # Randomly shuffle data
    shuffled_X, shuffled_y = randomize_input(iris_X, iris_y, rng)
    # Split data into train and test sets
    train_split = round(0.8 * shuffled_X.shape[0])
    train_inputs = shuffled_X[0:train_split].reshape(120, 4, 1)
    train_targets = shuffled_y[0:train_split]

    test_inputs = shuffled_X[train_split:].reshape(30, 4, 1)
    test_targets = shuffled_y[train_split:]
    # One hot encode targets e.g. [1,0,0]
    train_targets_enc = np.array(one_hot_encode(train_targets)).reshape(120, 3, 1)
    test_targets_enc = np.array(one_hot_encode(test_targets)).reshape(30, 3, 1)

    # batch inputs (I think these are technically "mini-batches" since parameters are updates n/batch_size times.
    batch_size = 1
    train_inputs, train_targets_enc = batch_input(
        [train_inputs, train_targets_enc], batch_size)

    # initialize layers from above
    params = initialize_layers(num_hlayers=4, num_weights=16, input_size=4, output_size=3, batch_size=batch_size)

    # create gradient function with autograd by passing in cost function
    pass_grad = elementwise_grad(cross_entropy)

    train_loop(pass_grad, params, train_inputs, train_targets_enc, 10, 0.01)

    test_loop(params, test_inputs, test_targets_enc)

