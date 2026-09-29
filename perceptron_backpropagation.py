import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1. Define functions
# ============================================================

def perceptron_feedforward(x1, x2, w1, w2, wb, b):

    activity = w1 * x1 + w2 * x2 + wb * b

    if activity > 0:
        return 1
    else:
        return 0


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):

    return output * (1 - output)


# ============================================================
# Part 2. Generate a linearly separable dataset
# ============================================================

np.random.seed(1)

num_samples = 100


# Generate random x and y values
x_data = np.random.uniform(-2, 2, num_samples)
y_data = np.random.uniform(-2, 2, num_samples)


# Define the true boundary
slope = 0.8
intercept = 0.2


# Store the correct answers
correct_answer = np.zeros(num_samples)


for i in range(num_samples):

    yline = slope * x_data[i] + intercept

    if y_data[i] > yline:
        correct_answer[i] = 1
    else:
        correct_answer[i] = 0


# ============================================================
# Part 3. Train a perceptron on the linear dataset
# ============================================================

w1 = 0.5
w2 = 0.5
wb = 0.5

b = 1

learning_constant = 0.01


linear_errors = []


for epoch in range(1000):

    for i in range(num_samples):

        x1 = x_data[i]
        x2 = y_data[i]

        target = correct_answer[i]

        guess = perceptron_feedforward(
            x1,
            x2,
            w1,
            w2,
            wb,
            b
        )

        # Single-sample error follows the classroom definition.
        error = target - guess

        if error != 0:

            w1 = w1 + learning_constant * x1 * error
            w2 = w2 + learning_constant * x2 * error
            wb = wb + learning_constant * b * error


    # Evaluate all samples with fixed weights after training this epoch.
    # Single-sample error is target - guess. For binary labels,
    # sum(abs(error)) counts misclassifications without cancellation.
    number_of_errors = 0

    for i in range(num_samples):

        guess = perceptron_feedforward(
            x_data[i], y_data[i], w1, w2, wb, b
        )
        error = correct_answer[i] - guess
        number_of_errors += abs(error)


    linear_errors.append(number_of_errors)


    if number_of_errors == 0:
        break


print("============================================")
print("PERCEPTRON: LINEAR CLASSIFICATION")
print("============================================")

print("Training stopped at epoch:", epoch + 1)

print("w1 =", w1)
print("w2 =", w2)
print("wb =", wb)

print("Final number of errors =", number_of_errors)


# ============================================================
# Part 4. Plot the linear classification result
# ============================================================

x_range = np.linspace(-2, 2, 100)


# True boundary
true_line = slope * x_range + intercept


# Perceptron boundary
learned_line = -(w1 * x_range + wb * b) / w2


# Find class 0 and class 1
class_0 = correct_answer == 0
class_1 = correct_answer == 1


plt.figure(figsize=(12, 5), layout="constrained")
plt.subplot(1, 2, 1)


# Class 0
plt.scatter(
    x_data[class_0],
    y_data[class_0],
    marker="x",
    color="black",
    label="Class 0"
)


# Class 1
plt.scatter(
    x_data[class_1],
    y_data[class_1],
    marker="o",
    facecolors="none",
    edgecolors="black",
    label="Class 1"
)


# True boundary
plt.plot(
    x_range,
    true_line,
    color="black",
    label="True boundary"
)


# Learned boundary
plt.plot(
    x_range,
    learned_line,
    "--",
    color="black",
    label="Perceptron boundary"
)


plt.xlabel("x1")
plt.ylabel("x2")

plt.title(
    "Perceptron on a Linearly Separable Dataset"
)

plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(linear_errors) + 1), linear_errors, color="black")
plt.xlabel("Epoch")
plt.ylabel("Misclassified samples after epoch")
plt.title("Perceptron learning: linear dataset")
plt.ylim(bottom=0)

plt.show()


# ============================================================
# Part 5. Generate a larger XOR dataset
# ============================================================

np.random.seed(2)


# Number of samples around each XOR corner
samples_per_group = 50


# Amount of random variation
noise = 0.08


# ------------------------------------------------------------
# Group 1: around (0, 0)
# Class 0
# ------------------------------------------------------------

group_00 = np.column_stack((

    np.random.normal(
        0,
        noise,
        samples_per_group
    ),

    np.random.normal(
        0,
        noise,
        samples_per_group
    )
))


# ------------------------------------------------------------
# Group 2: around (0, 1)
# Class 1
# ------------------------------------------------------------

group_01 = np.column_stack((

    np.random.normal(
        0,
        noise,
        samples_per_group
    ),

    np.random.normal(
        1,
        noise,
        samples_per_group
    )
))


# ------------------------------------------------------------
# Group 3: around (1, 0)
# Class 1
# ------------------------------------------------------------

group_10 = np.column_stack((

    np.random.normal(
        1,
        noise,
        samples_per_group
    ),

    np.random.normal(
        0,
        noise,
        samples_per_group
    )
))


# ------------------------------------------------------------
# Group 4: around (1, 1)
# Class 0
# ------------------------------------------------------------

group_11 = np.column_stack((

    np.random.normal(
        1,
        noise,
        samples_per_group
    ),

    np.random.normal(
        1,
        noise,
        samples_per_group
    )
))


# Combine all four groups
xor_inputs = np.vstack((

    group_00,
    group_01,
    group_10,
    group_11

))


# Create the answers
xor_answers = np.concatenate((

    np.zeros(samples_per_group),

    np.ones(samples_per_group),

    np.ones(samples_per_group),

    np.zeros(samples_per_group)

))


# ============================================================
# Part 6. Plot the XOR dataset
# ============================================================

class_0 = xor_answers == 0
class_1 = xor_answers == 1


plt.figure(figsize=(6, 6))


plt.scatter(
    xor_inputs[class_0, 0],
    xor_inputs[class_0, 1],
    marker="x",
    color="black",
    label="Class 0"
)


plt.scatter(
    xor_inputs[class_1, 0],
    xor_inputs[class_1, 1],
    marker="o",
    facecolors="none",
    edgecolors="black",
    label="Class 1"
)


plt.xlabel("x1")
plt.ylabel("x2")

plt.title("XOR Dataset")

plt.legend()

plt.show()


# ============================================================
# Part 7. Train a perceptron on XOR
# ============================================================

w1 = 0.5
w2 = 0.5
wb = 0.5

b = 1

learning_constant = 0.01


xor_perceptron_errors = []


for epoch in range(100):


    for i in range(len(xor_inputs)):

        x1 = xor_inputs[i, 0]
        x2 = xor_inputs[i, 1]

        target = xor_answers[i]


        guess = perceptron_feedforward(
            x1,
            x2,
            w1,
            w2,
            wb,
            b
        )


        # Single-sample error follows the classroom definition.
        error = target - guess


        if error != 0:

            w1 = w1 + learning_constant * x1 * error
            w2 = w2 + learning_constant * x2 * error
            wb = wb + learning_constant * b * error


    # Evaluate all samples with fixed weights after training this epoch.
    # Single-sample error is target - guess. For binary labels,
    # sum(abs(error)) is the total number of classification errors.
    # Counting updates during grouped XOR training is not this metric.
    number_of_errors = 0

    for i in range(len(xor_inputs)):

        guess = perceptron_feedforward(
            xor_inputs[i, 0], xor_inputs[i, 1], w1, w2, wb, b
        )
        error = xor_answers[i] - guess
        number_of_errors += abs(error)


    xor_perceptron_errors.append(
        number_of_errors
    )


# ============================================================
# Part 8. Evaluate the perceptron on the XOR training data
# ============================================================

number_correct = 0


for i in range(len(xor_inputs)):

    x1 = xor_inputs[i, 0]
    x2 = xor_inputs[i, 1]


    prediction = perceptron_feedforward(
        x1,
        x2,
        w1,
        w2,
        wb,
        b
    )


    if prediction == xor_answers[i]:

        number_correct += 1


perceptron_accuracy = (
    number_correct
    / len(xor_inputs)
    * 100
)


print("\n============================================")
print("PERCEPTRON: XOR")
print("============================================")

print(
    "Correct predictions:",
    number_correct,
    "/",
    len(xor_inputs)
)

print(
    "Training accuracy:",
    round(perceptron_accuracy, 2),
    "%"
)


# ============================================================
# Part 9. Plot the perceptron boundary on XOR
# ============================================================

x_range = np.linspace(
    -0.3,
    1.3,
    100
)


perceptron_line = -(
    w1 * x_range
    + wb * b
) / w2


plt.figure(figsize=(6, 6))


plt.scatter(
    xor_inputs[class_0, 0],
    xor_inputs[class_0, 1],
    marker="x",
    color="black",
    label="Class 0"
)


plt.scatter(
    xor_inputs[class_1, 0],
    xor_inputs[class_1, 1],
    marker="o",
    facecolors="none",
    edgecolors="black",
    label="Class 1"
)


plt.plot(
    x_range,
    perceptron_line,
    "--",
    color="black",
    label="Perceptron boundary"
)


plt.xlabel("x1")
plt.ylabel("x2")

plt.title(
    "A Single Perceptron Cannot Separate XOR"
)

plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)

plt.legend()

plt.show()


# ============================================================
# Part 10. Plot perceptron error during XOR training
# ============================================================

plt.figure(figsize=(7, 5), layout="constrained")


plt.plot(
    range(1, len(xor_perceptron_errors) + 1),
    xor_perceptron_errors,
    color="black"
)


plt.xlabel("Epoch")

plt.ylabel(
    "Misclassified samples after epoch"
)

plt.title(
    "Perceptron Training on XOR"
)

plt.show()


# ============================================================
# Part 11. Initialize a 2-2-1 neural network
# ============================================================

# Preserve the trained XOR perceptron before reusing weight names.
perceptron_w1 = w1
perceptron_w2 = w2
perceptron_bias = wb * b


np.random.seed(3)


# Input -> hidden neuron 1
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)


# Input -> hidden neuron 2
w3 = np.random.uniform(-1, 1)
w4 = np.random.uniform(-1, 1)


# Hidden neurons -> output neuron
w5 = np.random.uniform(-1, 1)
w6 = np.random.uniform(-1, 1)


# Biases
b1 = np.random.uniform(-1, 1)
b2 = np.random.uniform(-1, 1)
b3 = np.random.uniform(-1, 1)


learning_constant = 0.5


loss_history = []


# ============================================================
# Part 12. Train neural network using backpropagation
# ============================================================

for epoch in range(100):

    for i in range(len(xor_inputs)):

        x1 = xor_inputs[i, 0]
        x2 = xor_inputs[i, 1]

        target = xor_answers[i]


        # ====================================================
        # Feedforward
        # ====================================================

        h1_input = (
            w1 * x1
            + w2 * x2
            + b1
        )


        h1 = sigmoid(
            h1_input
        )


        h2_input = (
            w3 * x1
            + w4 * x2
            + b2
        )


        h2 = sigmoid(
            h2_input
        )


        output_input = (
            w5 * h1
            + w6 * h2
            + b3
        )


        output = sigmoid(
            output_input
        )


        # Single-sample loss: 0.5 * (target - output) ** 2.
        # Its derivative with respect to output is output - target.

        # ====================================================
        # Backpropagation: output neuron
        # ====================================================

        delta_output = (
            (output - target)
            * sigmoid_derivative(output)
        )


        gradient_w5 = (
            delta_output
            * h1
        )


        gradient_w6 = (
            delta_output
            * h2
        )


        gradient_b3 = (
            delta_output
        )


        # ====================================================
        # Backpropagation: hidden neuron 1
        # ====================================================

        delta_h1 = (
            delta_output
            * w5
            * sigmoid_derivative(h1)
        )


        gradient_w1 = (
            delta_h1
            * x1
        )


        gradient_w2 = (
            delta_h1
            * x2
        )


        gradient_b1 = (
            delta_h1
        )


        # ====================================================
        # Backpropagation: hidden neuron 2
        # ====================================================

        delta_h2 = (
            delta_output
            * w6
            * sigmoid_derivative(h2)
        )


        gradient_w3 = (
            delta_h2
            * x1
        )


        gradient_w4 = (
            delta_h2
            * x2
        )


        gradient_b2 = (
            delta_h2
        )


        # ====================================================
        # Update weights
        # ====================================================

        w1 = (
            w1
            - learning_constant
            * gradient_w1
        )


        w2 = (
            w2
            - learning_constant
            * gradient_w2
        )


        w3 = (
            w3
            - learning_constant
            * gradient_w3
        )


        w4 = (
            w4
            - learning_constant
            * gradient_w4
        )


        w5 = (
            w5
            - learning_constant
            * gradient_w5
        )


        w6 = (
            w6
            - learning_constant
            * gradient_w6
        )


        b1 = (
            b1
            - learning_constant
            * gradient_b1
        )


        b2 = (
            b2
            - learning_constant
            * gradient_b2
        )


        b3 = (
            b3
            - learning_constant
            * gradient_b3
        )


    # Evaluate mean loss with fixed weights after training this epoch.
    # Averaging keeps the metric independent of the number of samples.
    total_loss = 0

    for i in range(len(xor_inputs)):

        x1 = xor_inputs[i, 0]
        x2 = xor_inputs[i, 1]
        target = xor_answers[i]

        h1 = sigmoid(w1 * x1 + w2 * x2 + b1)
        h2 = sigmoid(w3 * x1 + w4 * x2 + b2)
        output = sigmoid(w5 * h1 + w6 * h2 + b3)

        loss = 0.5 * (target - output) ** 2
        total_loss += loss

    mean_loss = total_loss / len(xor_inputs)
    loss_history.append(mean_loss)


# ============================================================
# Part 13. Evaluate neural network on the XOR training data
# ============================================================

number_correct = 0


for i in range(len(xor_inputs)):

    x1 = xor_inputs[i, 0]
    x2 = xor_inputs[i, 1]

    target = xor_answers[i]


    # Hidden neuron 1
    h1 = sigmoid(
        w1 * x1
        + w2 * x2
        + b1
    )


    # Hidden neuron 2
    h2 = sigmoid(
        w3 * x1
        + w4 * x2
        + b2
    )


    # Output neuron
    output = sigmoid(
        w5 * h1
        + w6 * h2
        + b3
    )


    if output > 0.5:

        prediction = 1

    else:

        prediction = 0


    if prediction == target:

        number_correct += 1


neural_network_accuracy = (
    number_correct
    / len(xor_inputs)
    * 100
)


print("\n============================================")
print("NEURAL NETWORK + BACKPROPAGATION: XOR")
print("============================================")

print(
    "Correct predictions:",
    number_correct,
    "/",
    len(xor_inputs)
)

print(
    "Training accuracy:",
    round(neural_network_accuracy, 2),
    "%"
)


# ============================================================
# Part 14. Plot neural network loss
# ============================================================

plt.figure(figsize=(7, 5), layout="constrained")


plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    color="black"
)


plt.xlabel("Epoch")
plt.ylabel(r"Mean loss after epoch: $\frac{1}{N}\sum_i\frac{1}{2}(t_i-o_i)^2$")

plt.title(
    "Backpropagation Training on XOR"
)

plt.show()


# ============================================================
# Part 15. Compare XOR decision regions in the input space
# ============================================================

# A dense grid shows predictions between the training samples.
grid_values = np.linspace(-0.3, 1.4, 300)
grid_x1, grid_x2 = np.meshgrid(grid_values, grid_values)

perceptron_scores = (
    perceptron_w1 * grid_x1
    + perceptron_w2 * grid_x2
    + perceptron_bias
)

grid_h1 = sigmoid(w1 * grid_x1 + w2 * grid_x2 + b1)
grid_h2 = sigmoid(w3 * grid_x1 + w4 * grid_x2 + b2)
grid_output = sigmoid(w5 * grid_h1 + w6 * grid_h2 + b3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

for ax, scores, threshold, title in zip(
    axes,
    [perceptron_scores, grid_output],
    [0, 0.5],
    [
        f"Perceptron: {perceptron_accuracy:.1f}% training accuracy",
        f"2-2-1 network: {neural_network_accuracy:.1f}% training accuracy"
    ]
):

    ax.contourf(
        grid_x1, grid_x2, (scores > threshold).astype(int),
        levels=[-0.5, 0.5, 1.5], colors=["white", "0.85"]
    )
    ax.contour(
        grid_x1, grid_x2, scores, levels=[threshold],
        colors="black", linewidths=1.5
    )
    ax.scatter(
        xor_inputs[class_0, 0], xor_inputs[class_0, 1],
        marker="x", color="black", label="True class 0"
    )
    ax.scatter(
        xor_inputs[class_1, 0], xor_inputs[class_1, 1],
        marker="o", facecolors="none", edgecolors="black",
        label="True class 1"
    )
    ax.set(xlabel="x1", ylabel="x2", title=title,
           xlim=(-0.3, 1.4), ylim=(-0.3, 1.4))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("XOR decision regions: white = predicted 0; gray = predicted 1")
plt.show()


# ============================================================
# Part 16. See what the hidden layer learns
# ============================================================

hidden_1 = sigmoid(w1 * xor_inputs[:, 0] + w2 * xor_inputs[:, 1] + b1)
hidden_2 = sigmoid(w3 * xor_inputs[:, 0] + w4 * xor_inputs[:, 1] + b2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

for ax, horizontal, vertical, title, xlabel, ylabel in zip(
    axes,
    [xor_inputs[:, 0], hidden_1],
    [xor_inputs[:, 1], hidden_2],
    ["Original input space", "Learned hidden-layer space"],
    ["x1", "h1"],
    ["x2", "h2"]
):

    ax.scatter(
        horizontal[class_0], vertical[class_0],
        marker="x", color="black", label="Class 0"
    )
    ax.scatter(
        horizontal[class_1], vertical[class_1],
        marker="o", facecolors="none", edgecolors="black", label="Class 1"
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

axes[0].set(xlim=(-0.3, 1.4), ylim=(-0.3, 1.4))
axes[1].set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

# output > 0.5 is equivalent to w5*h1 + w6*h2 + b3 > 0.
# Draw its straight boundary without dividing by a possibly zero weight.
hidden_values = np.linspace(-0.05, 1.05, 300)
hidden_grid_1, hidden_grid_2 = np.meshgrid(hidden_values, hidden_values)
hidden_scores = w5 * hidden_grid_1 + w6 * hidden_grid_2 + b3
axes[1].contour(
    hidden_grid_1, hidden_grid_2, hidden_scores,
    levels=[0], colors="black", linewidths=1.5
)
axes[1].text(
    0.97, 0.70, "Output boundary:\nw5*h1 + w6*h2 + b3 = 0",
    transform=axes[1].transAxes, fontsize=9, ha="right"
)

fig.suptitle("The hidden layer transforms XOR into a linearly separable representation")
plt.show()


# ============================================================
# Part 17. Final comparison
# ============================================================

print("\n============================================")
print("FINAL COMPARISON")
print("============================================")


print(
    "Perceptron training accuracy on XOR:",
    round(perceptron_accuracy, 2),
    "%"
)


print(
    "Neural network training accuracy on XOR:",
    round(neural_network_accuracy, 2),
    "%"
)
print("Final neural network mean loss:", round(loss_history[-1], 6))
print("Accuracy is measured on the training data, not unseen data.")

# ============================================================
# Part 18. Define a combined hidden activation
# ============================================================

# Both hidden neurons use this SAME activation; the network is still 2-2-1.
# Fixed mixture: no extra trainable parameters and no maximum selection.
# References for the component definitions:
# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Tanh.html
# https://docs.pytorch.org/docs/main/generated/torch.nn.modules.activation.LeakyReLU.html
mix_weight = 0.5
negative_slope = 0.1


def leaky_relu(z):
    return np.where(z >= 0, z, negative_slope * z)


def combined_activation(z):
    return mix_weight * np.tanh(z) + (1 - mix_weight) * leaky_relu(z)


def combined_derivative(z):
    # At z=0, use the right derivative for the Leaky ReLU component.
    leaky_gradient = np.where(z >= 0, 1.0, negative_slope)
    return (
        mix_weight * (1 - np.tanh(z) ** 2)
        + (1 - mix_weight) * leaky_gradient
    )


def hidden_activation(z, kind):
    if kind == "sigmoid":
        return sigmoid(z)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "leaky_relu":
        return leaky_relu(z)
    if kind == "combined":
        return combined_activation(z)
    raise ValueError("Unknown activation: " + kind)


def hidden_derivative(z, output, kind):
    if kind == "sigmoid":
        return output * (1 - output)
    if kind == "tanh":
        return 1 - output ** 2
    if kind == "leaky_relu":
        return np.where(z >= 0, 1.0, negative_slope)
    if kind == "combined":
        # The input z is needed: output*(1-output) is NOT valid here.
        return combined_derivative(z)
    raise ValueError("Unknown activation: " + kind)


def activation_forward(inputs, weights, kind):
    # Same parameter order as the original code: w1,...,w6,b1,b2,b3.
    a1, a2, a3, a4, a5, a6, c1, c2, c3 = weights
    x1 = inputs[..., 0]
    x2 = inputs[..., 1]
    z1 = a1 * x1 + a2 * x2 + c1
    z2 = a3 * x1 + a4 * x2 + c2
    h1 = hidden_activation(z1, kind)
    h2 = hidden_activation(z2, kind)
    output = sigmoid(a5 * h1 + a6 * h2 + c3)
    return z1, z2, h1, h2, output


def sample_gradients(inputs, target, weights, kind):
    z1, z2, h1, h2, output = activation_forward(inputs, weights, kind)
    delta_output = (output - target) * output * (1 - output)
    # Compute every gradient with the OLD output-layer weights.
    delta_h1 = delta_output * weights[4] * hidden_derivative(z1, h1, kind)
    delta_h2 = delta_output * weights[5] * hidden_derivative(z2, h2, kind)
    x1, x2 = inputs
    return np.array([
        delta_h1 * x1, delta_h1 * x2,
        delta_h2 * x1, delta_h2 * x2,
        delta_output * h1, delta_output * h2,
        delta_h1, delta_h2, delta_output
    ])


def train_activation_model(inputs, targets, kind, seed, epochs, rate):
    # Local generator reproduces the original np.random.seed initialization,
    # without changing the global random state.
    rng = np.random.RandomState(seed)
    weights = rng.uniform(-1, 1, 9)
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Same fixed sample order for all compared activations.
        for i in range(len(inputs)):
            gradients = sample_gradients(inputs[i], targets[i], weights, kind)
            weights = weights - rate * gradients

        # Freeze weights and evaluate the ENTIRE dataset after the epoch.
        output = activation_forward(inputs, weights, kind)[-1]
        losses.append(np.mean(0.5 * (targets - output) ** 2))
        accuracies.append(np.mean((output > 0.5) == targets) * 100)

    return weights, np.array(losses), np.array(accuracies)


# ============================================================
# Part 19. Train the new model under the original conditions
# ============================================================

comparison_epochs = len(loss_history)
comparison_rate = 0.5
comparison_seed = 3

combined_weights, combined_losses, combined_accuracies = train_activation_model(
    xor_inputs, xor_answers, "combined", comparison_seed,
    comparison_epochs, comparison_rate
)

# Preserve the original trained sigmoid model for comparison.
sigmoid_weights = np.array([w1, w2, w3, w4, w5, w6, b1, b2, b3])

print("\n============================================")
print("COMBINED HIDDEN ACTIVATION: XOR")
print("============================================")
print("Activation: 0.5*tanh(z) + 0.5*LeakyReLU(z), negative slope = 0.1")
print("Architecture: 2-2-1; output activation: sigmoid")
print("Epochs:", comparison_epochs, "Learning rate:", comparison_rate)
print("Training accuracy:", combined_accuracies[-1], "%")
print("Final mean loss:", combined_losses[-1])
print("This is an experimental activation, not a guaranteed improvement.")


# ============================================================
# Part 20. Compare activation shapes and training loss
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
z_values = np.linspace(-4, 4, 400)
axes[0].plot(z_values, np.tanh(z_values), "--", color="0.35", label="tanh")
axes[0].plot(z_values, leaky_relu(z_values), ":", color="0.35", label="Leaky ReLU")
axes[0].plot(z_values, combined_activation(z_values), color="black", label="Combined")
axes[0].set(xlabel="Pre-activation z", ylabel="Hidden activation", title="Two-function mixture")
axes[0].legend()

epoch_numbers = range(1, comparison_epochs + 1)
axes[1].plot(epoch_numbers, loss_history, "--", color="black", label="Sigmoid hidden layer")
axes[1].plot(epoch_numbers, combined_losses, color="black", label="Combined hidden layer")
axes[1].set(xlabel="Epoch", ylabel="Mean half-squared loss after epoch",
            title="Same seed, data order, learning rate and epochs")
axes[1].legend()
plt.show()


# ============================================================
# Part 21. Compare final decision regions
# ============================================================

comparison_grid = np.stack((grid_x1, grid_x2), axis=-1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

for ax, kind, weights, accuracy in zip(
    axes, ["sigmoid", "combined"],
    [sigmoid_weights, combined_weights],
    [neural_network_accuracy, combined_accuracies[-1]]
):
    output = activation_forward(comparison_grid, weights, kind)[-1]
    ax.contourf(grid_x1, grid_x2, (output > 0.5).astype(int),
                levels=[-0.5, 0.5, 1.5], colors=["white", "0.85"])
    if output.min() < 0.5 < output.max():
        ax.contour(grid_x1, grid_x2, output, levels=[0.5], colors="black")
    ax.scatter(xor_inputs[class_0, 0], xor_inputs[class_0, 1],
               marker="x", color="black", label="True class 0")
    ax.scatter(xor_inputs[class_1, 0], xor_inputs[class_1, 1],
               marker="o", facecolors="none", edgecolors="black", label="True class 1")
    ax.set(xlabel="x1", ylabel="x2", xlim=(-0.3, 1.4), ylim=(-0.3, 1.4),
           title=f"{kind}: {accuracy:.1f}% training accuracy")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
fig.suptitle("White = predicted 0; gray = predicted 1")
plt.show()


# ============================================================
# Part 22. Check sensitivity to initialization
# ============================================================

# Compare each component separately as well as their mixture.
# All settings are fixed in advance; no best-seed selection or retraining.
# This measures training performance on ONE dataset, not test accuracy.
# Set False when presenting if you do not want to wait for the repeated runs.
run_seed_comparison = True
comparison_seeds = range(10)
activation_kinds = ["sigmoid", "tanh", "leaky_relu", "combined"]
seed_results = {}

if run_seed_comparison:
    print("\nINITIALIZATION CHECK: seeds 0-9, same fixed training data")
    print("Activation        Mean accuracy    100% runs    Mean loss")

    for kind in activation_kinds:
        results = []
        for seed in comparison_seeds:
            _, losses, accuracies = train_activation_model(
                xor_inputs, xor_answers, kind, seed,
                comparison_epochs, comparison_rate
            )
            results.append([accuracies[-1], losses[-1]])
        seed_results[kind] = np.array(results)
        values = seed_results[kind]
        print(f"{kind:16s} {values[:, 0].mean():8.2f}%"
              f"       {np.sum(values[:, 0] == 100):2d}/{len(values)}"
              f"       {values[:, 1].mean():.6f}")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for kind, style in zip(activation_kinds, ["--", ":", "-.", "-"]):
        ax.plot(list(comparison_seeds), seed_results[kind][:, 0],
                style, marker="o", color="black", label=kind)
    ax.set(xlabel="Initialization seed", ylabel="Final training accuracy (%)",
           title="Initialization sensitivity on the same XOR training set")
    ax.set_xticks(list(comparison_seeds))
    ax.set_ylim(0, 105)
    ax.legend()
    plt.show()
