import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tools import load_iris


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        '''
        Here we define the layers of the network.
        My SimpleNetwork will take in 5-dimensional
        features and return a 3-dimensional output
        prediction. This could for example be used for
        the Iris dataset which has 4 different features
        dimensions and 3 different flower types.

        I apply softmax on the last layer to get a
        propability distribution over classes
        '''

        self.fc_1 = nn.Linear(4, 100)
        self.fc_2 = nn.Linear(100, 100)
        self.fc_3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        This method performs the forward pass,
        x is the input feature being passed through
        the network at the current time
        '''
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return self.softmax(x)


class IrisDataSet(Dataset):
    def __init__(self):
        '''
        A simple PyTorch dataset for the Iris data
        '''

        features, targets, self.classes = load_iris()
        # we first have to convert the numpy data to compatible
        # PyTorch data:
        # * Features should be of type float
        # * Class labels should be of type long
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        '''We always have to define this method
        so PyTorch knows how many items are in our dataset
        '''
        return self.features.shape[0]

    def __getitem__(self, i):
        '''We also have to define this method to tell
        PyTorch what the i-th element should be. In our
        case it's simply the i-th elements from both features
        and targets
        '''
        return self.features[i, :], self.targets[i]


def create_iris_data_loader():
    '''Another convinient thing in PyTorch is the dataloader
    It allows us to easily iterate over all the data in our
    dataset. We can also:
    * set a batch size. In short, setting a batch size > 1
    allows us to train on more than 1 sample at a time and this
    generally decreases training time
    * shuffe the data.
    '''
    dl = DataLoader(IrisDataSet(), batch_size=10, shuffle=True)
    return dl


def train_simple_model():
    # Set up the data
    ds = IrisDataSet()
    dl = DataLoader(ds, batch_size=10, shuffle=True)

    # Initialize the model
    model = SimpleNetwork()

    # Choose a loss metric, cross entropy is often used
    loss_metric = nn.CrossEntropyLoss()
    # Choose an optimizer and connect to model weights
    # Often the Adam optimizer is used
    optimizer = torch.optim.Adam(model.parameters())

    num_steps = 0
    loss_values = []
    # THE TRAINING LOOP
    # we will do 50 epochs (i.e. we will train on all data 50 times)
    for epoch in range(50):
        for (feature, target) in dl:
            num_steps += 1
            # We have to do the following before each
            # forward pass to clear the gradients that
            # we have calculated in the previous step
            optimizer.zero_grad()
            # the prediction output of the network
            # will be a [10 x 3] tensor where out[i,:]
            # represent the class prediction probabilities
            # for sample i.
            out = model(feature)
            # Calculate the loss for the current batch
            loss = loss_metric(out, target)
            # To perform the backward propagation we do:
            loss.backward()
            # The optimizer then tunes the weights of the model
            optimizer.step()

            loss_values.append(loss.mean().item())

    plt.plot(loss_values)
    plt.title('Loss as a function of training steps')
    plt.show()


if __name__ == '__main__':
    train_simple_model()
