import numpy as np
import math
import random
import argparse
import json
import sys
import os

class Activation:
    
    @staticmethod
    def sigmoid(x, derivative=False):
        if not derivative:
            return 1/(1+np.exp(-x))
        return x * (1 - x)
    @staticmethod
    def relu(x, derivative=False):
        pass
    @staticmethod
    def tan_h(x, derivative=False):
        if not derivative:
            return np.tanh(x)
        return 1 - np.tanh(x)**2
        
class Network:
    def __init__(self, size=[2,3,2], activation_func=Activation.tan_h, learning_rate=0.1, momentum=0, momentum_bool=False, sigmoid_last_bool=False):
        self.size = size
        self.weights = []
        self.biases = []
        self.data = []
        self.activation = activation_func
        self.learning_rate = learning_rate
        self.momentum = 0.1
        self.momentum_bool = momentum_bool
        self.sigmoid_last_bool = sigmoid_last_bool
        
        self.momentum = momentum
        for i in range(len(size)-1):
            self.weights.append(self.rand_matrix(size[i+1], size[i]))
            self.biases.append(self.rand_matrix(size[i+1], 1))
            
        self.momentum_weights = [np.zeros_like(w) for w in self.weights]
        self.momentum_bias = [np.zeros_like(w) for w in self.biases]
        
        
    """
	this method initialises a matrix
    """
    def rand_matrix(self, row, col):
        return np.random.rand(row, col)
    
    def feed_forward(self, inputs):
        if len(inputs)!=self.size[0]:
            print("Ooopps")
            
        current = np.array(inputs, dtype=np.float32).reshape(-1, 1)
        self.data = [current.copy()]
        
        for i in range(len(self.size)-1):

            current = np.dot(self.weights[i], current)
            current = current+self.biases[i]

            if i==len(self.weights) - 1 and self.sigmoid_last_bool:

                current = Activation.sigmoid(current)
            else:
                current = self.activation(current)
            
            self.data.append(current)
        return current

    def back_prop(self, outputs, targets):
        parsed = outputs
        target_matrix = targets

        errors = target_matrix - parsed

        
        if self.sigmoid_last_bool:
            gradient = Activation.sigmoid(parsed, derivative=True)
        else:
            gradient = self.activation(parsed, derivative=True)
        

        for i in range(len(self.size) - 2, -1, -1):
            gradient_term = gradient * errors * self.learning_rate

            if self.momentum_bool:
                self.momentum_weights[i] = self.momentum * self.momentum_weights[i] + np.dot(gradient_term, self.data[i].T)
                self.momentum_bias[i] = self.momentum * self.momentum_bias[i] + gradient_term

                self.weights[i] += self.momentum_weights[i]
                self.biases[i] += self.momentum_bias[i]
            else:
                self.weights[i] += np.dot(gradient_term, self.data[i].T)
                self.biases[i] += gradient_term

            errors = np.dot(self.weights[i].T, errors)

            gradient = self.activation(self.data[i], derivative=True)

        
    def train(self, inputs, epoch):
        rand_inputs = inputs.copy()
        for i in range(epoch):
            print(f"\rProgress: {i} out of {epoch}", end="", flush=True)
            random.shuffle(rand_inputs)
            for j in range(len(inputs)):
                output = self.feed_forward(rand_inputs[j][0].copy())
                self.back_prop(output, rand_inputs[j][1])

    def test(self, info):
        corr = 0
        total = len(info)
    
        for data, labels in info:
            output = self.feed_forward(data)
        
            # Ensure shapes match
            output_rounded = np.round(output)
            if output_rounded.shape != labels.shape:
                labels = labels.reshape(output_rounded.shape)
        
            if np.array_equal(output_rounded, labels):
                corr += 1

        accuracy = corr / total
        return {"accuracy": accuracy}

        

def read_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            size_numbers = np.array([int(x) for x in first_line.split()])

            data_labels = []

            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                # Labels
                label = int(parts[0])
                if label == 1:
                    label = np.array([1, 0], dtype=np.float32).reshape(-1, 1)
                else:
                    label = np.array([0, 1], dtype=np.float32).reshape(-1, 1)

                # Features
                data = np.array([float(x) for x in parts[1:]], dtype=np.float32, ndmin=1)

                # Standardize each row
                mean = data.mean()
                std = data.std() if data.std() > 0 else 1.0  # prevent division by zero
                data = ((data - mean) / std).reshape(-1, 1)

                data_labels.append((data, label))

        return size_numbers, data_labels

    except FileNotFoundError:
        print("File not found")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
def split_data(data):
    np.random.seed(8)

    # Shuffle indices
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Compute split points
    n = len(data)
    n1 = int(0.33 * n)
    n2 = int(0.33 * n)

    idx1 = indices[:n1]
    idx2 = indices[n1:n1 + n2]
    idx3 = indices[n1 + n2:]

    return [data[i] for i in idx1], [data[i] for i in idx2], [data[i] for i in idx3]

def main():
    parser = argparse.ArgumentParser(description="Run script with a configuration file.")
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    parser.add_argument("--cmdln", required=False, help="use this flag to run the code on the command line")
    args = parser.parse_args()

    # Load the config
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config: {args.config}")
        print(e)
        sys.exit(1)

    print(f"Running with configuration: {args.config}")
    print("Config contents:", config)
    activation = Activation.sigmoid
    results = {}
    if config.get("activation") == "tanh":
        activation = Activation.tan_h
    if config.get("momentum") == "1":
        momentum = True
    if config.get("sigmoid_last") == "1":
        sigmoid_last = True


    size_of_file, data_labels = read_data(os.path.join("data",config.get("data_file")))

    data1, data2, data3 = split_data(data_labels)

    test = Network(size=config.get("size"), activation_func=activation, momentum_bool=momentum, sigmoid_last_bool=sigmoid_last)    
    test.train(data2+data3, config.get("epochs"))
    results["standard"] = test.test(data1)
        
    
    
    if not args.cmdln:
        output_dir = "logs"
        os.makedirs(output_dir, exist_ok=True)
    
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(results)
        
    

if __name__ == "__main__":
    main()