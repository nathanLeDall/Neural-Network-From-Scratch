# Neural Network From Scratch

A feed-forward neural network implemented from scratch in Python using **NumPy**, without TensorFlow, PyTorch, Keras, or other machine-learning frameworks.

This project explores the mathematics and core mechanics behind neural networks by manually implementing forward propagation, backpropagation, weight and bias updates, activation functions, training, classification, and model evaluation.

## Features

* Neural network implementation built from scratch
* Configurable network layer architecture
* Feed-forward propagation
* Backpropagation
* Weight and bias updates
* Mean Squared Error (MSE) loss calculation
* Multiple activation functions:

  * Sigmoid
  * ReLU
  * Tanh
  * Softmax
* Optional Softmax output layer
* Momentum update support
* Input data normalization
* Randomized training data
* Train/test data splitting
* Classification accuracy evaluation
* Training loss visualization with Matplotlib
* JSON-based experiment configuration
* Support for multiple datasets

## Technologies

* **Python**
* **NumPy**
* **Matplotlib**

The neural-network calculations themselves are implemented manually rather than through a machine-learning framework.

## How It Works

The network consists of an input layer, one or more hidden layers, and an output layer.

For every training example, the program performs:

1. **Feed Forward** — input values are passed through each layer using matrix multiplication, biases, and activation functions.
2. **Loss Calculation** — the predicted output is compared against the expected output.
3. **Backpropagation** — the error is propagated backward through the network.
4. **Parameter Updates** — weights and biases are adjusted based on the calculated gradients.
5. **Evaluation** — predictions are compared with the correct labels to calculate classification accuracy.

Training loss is recorded across epochs and visualized using Matplotlib.

## Project Structure

```text
Neural-Network-From-Scratch/
│
├── data/                 # Training datasets
├── logs/                 # Generated experiment results
├── .github/workflows/    # GitHub Actions configuration
│
├── main.py               # Neural network implementation
├── input_test.py         # Input/testing utilities
├── config_1.json         # Experiment configuration
├── config_2.json
├── config_3.json
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nathanLeDall/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires:

```text
numpy
matplotlib
```

## Running the Network

Experiments are controlled through JSON configuration files.

For example:

```bash
python main.py --config config_1.json --cmdln 1
```

A configuration can look like:

```json
{
  "size": [16, 6, 2],
  "learning_rate": 0.0001,
  "epochs": 1000,
  "data_file": ["L30fft16.out"],
  "activation": "tanh",
  "momentum": "1",
  "softmax_last": "0"
}
```

## Configuration

### `size`

Defines the architecture of the neural network.

```json
"size": [16, 6, 2]
```

This represents:

```text
16 Input Neurons
       ↓
6 Hidden Neurons
       ↓
2 Output Neurons
```

Additional hidden layers can be represented by adding more values to the array.

For example:

```json
"size": [16, 32, 16, 8, 2]
```

### `epochs`

Controls the number of times the network trains over the training data.

```json
"epochs": 1000
```

### `data_file`

Specifies the datasets located inside the `data/` directory.

```json
"data_file": ["L30fft16.out"]
```

Multiple datasets can also be supplied:

```json
"data_file": [
  "dataset1.out",
  "dataset2.out"
]
```

### `activation`

Controls the activation behavior used by the configured experiment.

Example:

```json
"activation": "tanh"
```

The implementation contains activation functions for:

```text
Sigmoid
ReLU
Tanh
Softmax
```

### `momentum`

Enables or disables the momentum update path.

```json
"momentum": "1"
```

Values:

```text
1 = Enabled
0 = Disabled
```

### `softmax_last`

Determines whether Softmax is applied to the final layer.

```json
"softmax_last": "1"
```

Values:

```text
1 = Enabled
0 = Disabled
```

## Data Processing

Before training, each input sample is normalized using its mean and standard deviation:

```text
x' = (x - mean) / standard deviation
```

The dataset is randomized and divided into three groups that are used across training and evaluation runs.

For classification, labels are represented using two-element vectors such as:

```text
[1, 0]
[0, 1]
```

The predicted class is selected using the output neuron with the highest activation value.

## Backpropagation

Backpropagation is implemented directly using NumPy operations.

The network:

* Calculates prediction error
* Calculates the activation derivative
* Determines the gradient for each layer
* Propagates errors backward through the network
* Updates weights
* Updates biases

This makes the project useful for understanding what machine-learning frameworks normally handle automatically.

## Training Visualization

Training loss is collected after each epoch and plotted using Matplotlib.

The generated graph displays:

```text
X-axis: Epoch
Y-axis: Loss
```

This makes it possible to visualize how the network's error changes throughout training.

## Evaluation

The network evaluates classification performance by comparing:

```python
np.argmax(prediction)
```

with:

```python
np.argmax(expected_label)
```

Accuracy is calculated as:

```text
Correct Predictions / Total Predictions
```

When the program is run without command-line output enabled, results can be written to:

```text
logs/results.json
```

## Purpose

The goal of this project was to build a neural network without relying on high-level machine-learning libraries in order to develop a deeper understanding of:

* Artificial neural networks
* Linear algebra
* Matrix operations
* Forward propagation
* Backpropagation
* Gradient-based learning
* Activation functions
* Data normalization
* Classification
* Model evaluation

Rather than treating a neural network as a black box, the project implements the underlying training process directly.

## Author

**Nathan Le Dall**

GitHub: [@nathanLeDall](https://github.com/nathanLeDall)
