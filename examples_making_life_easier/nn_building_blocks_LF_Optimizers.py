"""
higher level of building blocks of NN architecture,
popular optimization, algorithms and loss functions tells us how far we are from actual prediction (error)
Loss Functions:

    Regression Problems:
        - MSELoss (Mean Squared Error) - loss for regression problems

    Binary Classification Problems:
        - BCELoss (Binary Cross Entropy) - loss for binary classification problems, single prob value usually output of sigmoid
        - BCEWithLogits -  similar to BCELoss except accepts raw score and applies sigmoid itself.

    MultiClass Classification Problems:
        -CrossEntropyLoss: expects raw scores and applies LogSoftMax internally
        -NLLLoss: expects to have log probabilities as input

Optimizers:
    can make a difference sometimes in converging dynamics.

    SGD: vanilla stochastic gradient descent algo with momentum extension
    RMSprop: optimizer proposed by hinton
    Adagrad: Adaptive Gradient optimizers
"""

from torch import nn
from torch import FloatTensor

def simple_linear_model_example():
    """
    illustrates linear model and building blocks of NN architecture
    linear_nn is FFNN (Feed Forward Neural Network) in example contains 2 features input and 5 features output
    Methods of nn module:
        - parameters(): returns iterator of all variable requiring gradient computation (weights)
        - zero_grad(): initializes gradient of all parameters to zero
        - to(device): moves modules to given device
        - state_dict(): returns dictionary with all module parameters, useful for model serialization
        - load_state_dict(): initializes module parameters with state_dictionary
    :return: None
    """
    linear_nn = nn.Linear(2, 5)
    temp_var = FloatTensor([1, 2])
    print("input shape: ", temp_var.shape)
    print("output shape: ", linear_nn(temp_var).shape, " output: ", linear_nn(temp_var))

def nn_sequential_example():
    """
    example of sequential block in neural network using pytorch
    :return: None
    """
    # 3 layer neural network with ReLU non-linearity, dropout and softmax on output along dim 1 (dim 0 is batch samples)
    s = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Dropout(0.3),
        nn.Softmax(dim=1),
    )
    print("Network: \n", s)
    temp_var = FloatTensor([[1, 2]])
    out_network = s(temp_var)
    print("network input: ", temp_var)
    print("Network Out: ", out_network)

class CustomNeuralNetworkModule(nn.Module):
    """
    creating our own neural network module
    """
    def __init__(self, input_dim, out_dim, dropout_rate=0.3):
        super(CustomNeuralNetworkModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, out_dim),
            nn.Dropout(p=dropout_rate),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        """
        we override forward function with our data transformation
        :param x:
        :return:
        """
        return self.pipe(input)

def blueprint_training_loop_example():
    """
    contains blueprint of training loop
    :return:
    """
    # for batch_samples, batch_labels in iterate_batches(data, batch_size=32):
    #     batch_samples_t = torch.tensor(batch_samples)
    #     batch_labels_t = torch.tensor(batch_labels)
    #     out_t = net(batch_samples_t)
    #     loss_t = loss_function(out_t, batch_labels_t)
    #     loss_t.backward() # every tensor in computation graph remembers it parent.
    #     optimizer.step()
    #     optimizer.zero_grad()
    ...


if __name__ == '__main__':
    simple_linear_model_example()
    nn_sequential_example()
    # example of calling our own NN module
    print("example of our own neural network module overriding forward function")
    our_net = CustomNeuralNetworkModule(2, 3)
    temp_var = FloatTensor([[2, 3]])
    our_net_out = our_net(temp_var)
    print("network \n :", our_net)
    print("network input: ", temp_var)
    print("output output :", our_net_out)