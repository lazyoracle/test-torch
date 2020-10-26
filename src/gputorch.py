import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import getopt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cuda_check() -> bool:
    """check if GPU is available

    Returns
    -------
    bool
        True is GPU is being used
    """
    return torch.cuda.is_available()


def train(total_epochs: int, PATH: str) -> float:
    """Train a classifier on the CIFAR dataset

    Parameters
    ----------
    total_epochs : int
        total number of training epochs
    PATH : str
        path to save model

    Returns
    -------
    float
        final loss value
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    show_loss = 0.0

    for epoch in range(total_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                show_loss = running_loss / 2000
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, show_loss))
                running_loss = 0.0

    torch.save(net.state_dict(), PATH)

    return show_loss

def test(PATH: str) -> float:
    """test pre-trained model on test set

    Parameters
    ----------
    PATH : str
        path to pre-trained model

    Returns
    -------
    float
        accuracy of the model
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100 * correct / total)

def main(argv):
    PATH = "./model.pth"
    epochs = 2
    try:
        opts, args = getopt.getopt(argv, "hp:e:", ["path=", "epochs="])
    except getopt.GetoptError:
        print("train.py -p <path> -e <epochs>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("train.py -p <path> -e <epochs>")
            sys.exit(2)
        elif opt in ("-p", "--path"):
            PATH = str(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
    
    train(total_epochs = epochs, PATH = PATH)
    test(PATH = PATH)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("train.py -p <path> -e <epochs>")
    else:
        main(sys.argv[1:])
