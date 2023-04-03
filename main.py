# Neural Network from numpy
import numpy
import numpy as np
from sklearn import datasets as ds
from tqdm import tqdm

# TODO add docstrings

def v_relu(v):  # activation function
    def relu(x):
        return 0 if x <= 0 else x

    return np.vectorize(relu)(v)


def dv_relu(v):  # derivative of v_relu
    def d_relu(x):
        return 0 if x <= 0 else 1

    return np.vectorize(d_relu)(v)


def weigh_and_bias(input_vector, weights, bias):
    weighted = np.matmul(input_vector, weights)
    return weighted + bias


def softmax(v):
    shift_to_avoid_instability = v - np.max(v)
    return np.exp(shift_to_avoid_instability) / np.sum(np.exp(shift_to_avoid_instability), axis=0)
    # Derivative if differentiating w/ respect to numerator is softmax(ith element)(1-softmax(ith elem)
    # Derivative if differentiating w/ respect to some other element, not the numerator, is -softmax(jth (diff wrt j)) * softmax(ith (numerator))


def cross_entropy(probabilities, labels):  # from softmax
    loss = -np.sum(labels * np.log(probabilities), axis=0)
    correct = 1 if np.argmax(probabilities) == np.argmax(labels) else 0
    return loss, correct
    # derivative wrt jth input is softmax(jth) - jth label

    # Not general solution, this code is specific to my layers
    # derivative of last layer is d(cost)/dSoftmax dSoftmax/dweights
    # Derivative of bias is d(c0st)/dSoftmax dSoftmax/dBias


def get_collapsed_kronecker(input_vector, size):
    return np.array([input_vector for i in range(size)]).reshape((size, size))


def forward_pass(layers, input):
    curr_vector = input
    input_vector_moments = [input]
    for i in range(len(layers) - 1):  # runs through all layers except the last which requires a different command
        weights, bias = layers[i] # made change 1 -> i
        w_sum_vector = weigh_and_bias(curr_vector, weights, bias)  # Weigh features and add bias
        input_vector_moments.append(w_sum_vector)
        activated_sum_vector = v_relu(w_sum_vector)
        curr_vector = activated_sum_vector

    final_sum_vector = weigh_and_bias(curr_vector, layers[-1][0], layers[-1][1])
    input_vector_moments.append(final_sum_vector)
    probabilities = softmax(final_sum_vector)

    return input_vector_moments, probabilities


def backpropagate(layers, sums, probabilities, labels):
    weight_derivatives = []
    bias_derivatives = []
    xent_and_smax_partial = np.array(probabilities - labels).transpose()
    for i in range(len(layers)):
        weight_partial = xent_and_smax_partial  # activation function and cost
        d_wrt_weights = get_collapsed_kronecker(sums[-1], len(weight_partial))
        weight_partial = np.matmul(d_wrt_weights, weight_partial)
        # size here is the # of rows of previous partial matrix

        bias_partial = xent_and_smax_partial

        for j in range(0, i+1):  # shift i values by 1 because we are negative indexing w/ j (no -0)
            if j != 0:
                weight_partial = np.matmul(layers[-j][0], weight_partial)
                weight_partial = dv_relu(weight_partial)
                d_wrt_weights = get_collapsed_kronecker(sums[-j-1], len(weight_partial))
                weight_partial = np.matmul(d_wrt_weights, weight_partial)

                bias_partial = np.matmul(layers[-j][0], bias_partial)  # weights of previous layer
                bias_partial = dv_relu(bias_partial)

        weight_derivatives.append(weight_partial)
        bias_derivatives.append(bias_partial)

    return weight_derivatives, bias_derivatives


def update_weights_and_biases(layers, weight_gradient, bias_gradient, step_size):
    new_layers = []
    for i in range(len(layers)):
        weights, bias = layers[i]
        weights = weights - (step_size * weight_gradient[-(i + 1)])
        bias = bias - (step_size * bias_gradient[-(i + 1)])
        new_layers.append((weights, bias))

    return new_layers


def average_gradients(gradients):
    w_grad, b_grad = gradients
    return np.mean(w_grad, axis=0), np.mean(b_grad, axis=0)


def train_loop(layers, input_vectors, labels, step_size, batch_size, epochs):  # TODO add loss tracker
    self_layers = layers
    w_grads = []
    b_grads = []
    for i in tqdm(range(epochs)):
        if len(w_grads) == batch_size:
            avg_w_grad, avg_b_grad = average_gradients((w_grads, b_grads))
            self_layers = update_weights_and_biases(self_layers, avg_w_grad, avg_b_grad,
                                                    step_size)
            w_grads = []
            b_grads = []
        sums, output, = forward_pass(self_layers, input_vectors[i])
        w_grad, b_grad = backpropagate(self_layers, sums, output, labels[i])
        w_grads.append(w_grad)
        b_grads.append(b_grad)

    if len(w_grads) != 0:
        avg_w_grad, avg_b_grad = average_gradients((w_grads, b_grads)) # one last one for the road if
        self_layers = update_weights_and_biases(self_layers, avg_w_grad, avg_b_grad,
                                                step_size)

    return self_layers


def one_hot_encode(labels, num_labels):  # NOTE U NEED TO TELL IT HOW MANY LABELS
    new_labels = []
    for label in labels:
        zero_array = np.zeros(num_labels)
        zero_array[label] = 1
        new_labels.append(zero_array)

    return new_labels


def randomize_input(features, labels):
    rng = np.random.default_rng()
    indices = np.arange(len(features))
    rng.shuffle(indices)

    return features[indices], labels[indices]


def test(layers, input, labels):
    assert len(input) == len(labels)
    loss_history = []
    accuracy = 0.0
    for i in tqdm(range(len(input))):
        _, probabilities = forward_pass(layers, input)
        loss, truth_value = cross_entropy(probabilities, labels)
        loss_history.append(loss)
        accuracy = accuracy + (truth_value - accuracy)/i

    return loss_history, accuracy


# get inputs
iris_X, iris_y = ds.load_iris(return_X_y=True)
shuffled_X, shuffled_y = randomize_input(iris_X, iris_y)
train_split = round(0.8 * shuffled_X.shape[0])
train_inputs = shuffled_X[0:train_split]
train_targets = shuffled_y[0:train_split]

test_inputs = shuffled_X[train_split:]
test_targets = shuffled_y[train_split:]

train_targets_enc = one_hot_encode(train_targets, 3)
test_targets_enc = one_hot_encode(test_targets,3)

# let's try 4 layers of size 16
layer1_w = numpy.ones((4, 16))
layer1_b = numpy.zeros((1, 16))

layer2_w = numpy.ones((16, 16))
layer2_b = numpy.zeros((1, 16))

layer3_w = numpy.ones((16, 16))
layer3_b = numpy.zeros((1, 16))

layer4_w = numpy.ones((16, 16))
layer4_b = numpy.zeros((1, 16))

output_layer_w = numpy.ones((16, 3 ))
output_layer_b = numpy.zeros((1, 3))

initial_layers = [(layer1_w, layer1_b), (layer2_w, layer2_b), (layer3_w, layer3_b), (layer4_w, layer4_b),
                  (output_layer_w, output_layer_b)]
# train
trained_layers = train_loop(initial_layers, train_inputs, train_targets_enc, 0.1, 5, 10)

# test

loss, accuracy = test(trained_layers, test_inputs, test_targets_enc)

print(accuracy)



# test

