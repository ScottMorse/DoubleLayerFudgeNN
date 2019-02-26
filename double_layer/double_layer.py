import numpy as np
from utils import *

class NeuralNetwork:

    def __init__(self,n_input,n_output,n_hidden,learn_rate):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.learn_rate = learn_rate
        self.hidden_weights = weight_matrix(n_input,n_hidden)
        self.output_weights = weight_matrix(n_hidden,n_output)

    def activate(self,weights,vector):
        return sigmoid(np.dot(weights,vector))

    def train(self,inputs,target_outputs):
        input_vector = transposed_vector(inputs)
        target_vector = transposed_vector(target_outputs)

        output_vector1 = self.activate(self.hidden_weights,input_vector)
        output_vector2 = self.activate(self.output_weights,output_vector1)

        output_diffs = target_vector - output_vector2
        output_error = output_diffs * output_vector2 * (1 - output_vector2)
        output_adjustments = self.learn_rate * np.dot(output_error,output_vector1.T)
        self.output_weights += output_adjustments

        hidden_error = np.dot(self.output_weights.T, output_diffs)
        hidden_adjustments = hidden_error * output_vector1 * (1 - output_vector1)
        self.hidden_weights += hidden_adjustments
    
    def think(self,inputs):
        input_vector = np.array(inputs, ndmin=2).T
        output_vector = sigmoid(np.dot(
                            self.output_weights,
                            sigmoid(np.dot(self.hidden_weights,
                                           input_vector))
                            )
                        )
        return output_vector


if __name__ == "__main__":

    INPUT_NODES = 3
    HIDDEN_NOTES = 4
    OUTPUT_NODES = 2
    LEARN_RATE = 0.1

    nn = NeuralNetwork(INPUT_NODES,OUTPUT_NODES,HIDDEN_NOTES,LEARN_RATE)

    print(f'\nHidden Starting Weights:\n{nn.hidden_weights}\n\nOutput Starting Weights:\n{nn.output_weights}')

    for _ in range(1000):
        nn.train([[1,2,3],[2,3,4],[3,4,5]],[[0,1],[1,2],[2,3]])

    print(f'\nHidden Ending Weights:\n{nn.hidden_weights}\n\nOutput Ending Weights:\n{nn.output_weights}')