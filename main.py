import autograd.numpy as np
from numpy.random import default_rng
from sklearn import datasets as ds
from autograd import grad


# TODO add docstrings
# TODO add type suggestions
# TODO Clean this up

def one_hot_encode(labels):
    new_labels = []
    num_labels = len(np.unique(labels))
    for label in labels:
        zero_array = np.zeros(num_labels)
        zero_array[label] = 1
        new_labels.append(zero_array)

    return new_labels


def randomize_input(features, labels, random):
    rng = random
    indices = np.arange(len(features))
    rng.shuffle(indices)

    return features[indices], labels[indices]


def v_relu(v):  # activation function
    new_array = []
    for i in range(len(v)):
        t = [0.0] if v[i] <= 0 else v[i]
        new_array.append(t)
    return np.array(new_array, ndmin=1)


def accuracy(layers, iv, label):
    guess = softmax(forward_pass(layers, iv))
    loss = cross_entropy(layers, guess, label, grad=False)
    return loss, np.mean(np.argmax(label) == np.argmax(guess))


def softmax(v):
    z = v - np.max(v)
    num = np.exp(z)
    denom = np.sum(num)
    return num / denom


def cross_entropy(layers, input, labels, grad=False):  # from softmax
    if grad:
        probs = forward_pass(layers, input)
        loss = -np.sum(labels * np.log(probs))
    else:
        loss = -np.sum(labels * np.log(input))
    return loss


def update_weights_and_biases(params, grad, step):
    for i in range(len(params)):
        w, b = params[i]
        w_grad, b_grad = grad[i]
        w -= step * w_grad
        b -= step * b_grad


def forward_pass(params, input_v):  # make sure everything is correct shape already
    curr_vector = input_v
    for w, b in params:
        weighted_sum = w @ curr_vector + b
        curr_vector = v_relu(weighted_sum)

    probs = softmax(weighted_sum)
    return probs


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

    def initialize_layers(num_hlayers, num_weights, input_size, output_size):  # TODO add batches
        # Input layer with special size requirements: num_weights x input_size
        layers = [(rng.normal(0, 2/input_size, (num_weights, input_size)), rng.standard_normal((num_weights, 1)))]
        # Hidden layers all share same size
        for i in range(num_hlayers - 2):
            layers.append((rng.normal(0, 2/num_weights, (num_weights, num_weights)), rng.standard_normal((num_weights, 1))))
        # Output layer has special size requirement: output_size x num_weights
        layers.append((rng.normal(0, 2/num_weights, (output_size, num_weights)), rng.standard_normal((output_size, 1))))
        return layers

    # initialize layers from above
    params = initialize_layers(num_hlayers=5, num_weights=10, input_size=4, output_size=3)

    # create gradient function with autograd by passing in cost function
    pass_grad = grad(cross_entropy)

    def train_loop(gradient_func, layers, input_vectors, labels, epochs, step_size):

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


    train_loop(pass_grad, params, train_inputs, train_targets_enc, 10, 0.01)

    def test_loop(layers, input, labels):
        m_loss = 0.0
        m_acc = 0.0
        for i in range(len(input)):
            test_loss, test_acc = accuracy(layers, input[i], labels[i])
            m_loss += (test_loss - m_loss) / (i + 1)
            m_acc += (test_acc - m_acc) / (i + 1)
        print("Validation loss: {l}, val acc: {a}".format(l=m_loss, a=m_acc))

    test_loop(params, test_inputs, test_targets_enc)