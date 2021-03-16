import torch
import torch.optim as optim
from CamVid import CamVidDataset


def train(model, trainloader, testloader, optimizer, device):
    pass

if __name__ == "__main__":
    # Define Hyperparameters
    device = torch.device('cuda:0')
    lr = 0.001
    # Load Data
    train_data = CamVidDataset(image_path='./CamVid/train', label_path='./CamVid/train_labels',transform=transform)
    test_data = CamVidDataset(image_path='./CamVid/val', label_path='./CamVid/vak_labels',transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)
    # Instantiate Model
    model = SegNet()
    #Define Optimizer
    criterion = 
    optimizer =
    # Train
    train(model, trainloader, testloader, optimizer, device)
    # Save Model to file
    with f = open('segnet.pkl', 'wb'):
        pickle.dump(model,f)
        f.close()
