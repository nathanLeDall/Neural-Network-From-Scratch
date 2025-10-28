import numpy as np
import math
import random
print ("test")

class Activation:
    
    @staticmethod
    def sigmoid(x, derivative=False):
        if not derivative:
            return 1/(1+math.pow(math.e,-1*x))
        return x * (1 - x)
        
class Network:
    def __init__(self, size=[2,3,2], activation_func=Activation.sigmoid, learning_rate=0.1):
        self.size = size
        self.weights = []
        self.biases = []
        self.data = []
        self.activation = activation_func
        self.learning_rate = learning_rate
        for i in range(len(size)-1):
            self.weights.append(self.rand_matrix(size[i+1], size[i]))
            self.biases.append(self.rand_matrix(size[i+1], 1))
            
        
        
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
            sigmoid_vec = np.vectorize(Activation.sigmoid)
            current = sigmoid_vec(current)
            
            self.data.append(current)
            #print (f" at {i} current ={current}")
        return current

    def back_prop(self, outputs, targets):
        parsed = outputs
        target_matrix = targets

        errors = target_matrix - parsed

        gradient = self.activation(parsed, derivative=True)

        for i in range(len(self.size) - 2, -1, -1):
            gradient_term = gradient * errors * self.learning_rate

            self.weights[i] += np.dot(gradient_term, self.data[i].T)
            self.biases[i] += gradient_term

            errors = np.dot(self.weights[i].T, errors)

            gradient = self.activation(self.data[i], derivative=True)

        
    def train(self, inputs, epoch):
        rand_inputs = inputs.copy()
        for i in range(epoch):
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
        print(f"Correct: {corr}/{total}  Accuracy: {accuracy:.2%}")
        return accuracy

        

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

 
if __name__ == "__main__":
    test = Network(size=[16,5,2])
    size_of_file, data_labels = read_data("data/L30fft16.out")
    test.test(data_labels)
    test.train(data_labels, 5000)
    test.test(data_labels)
    
    