import torch
import torch.optim as optim
from CamVid import CamVidDataset

device = torch.device('cuda:0')

def train(model, train_dl, test_dl, opt, loss_func, epochs):
    """ train model using using provided datasets, optimizer and loss function """
    train_loss = [0 for i in range(epochs)]
    test_loss  = [0 for i in range(epochs)]
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_func(model(xb), yb)
            train_loss[epoch] = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            losses, nums = zip(*[(loss_func(model(xb.to(device)),yb.to(device)).item(),len(xb.to(device))) for xb, yb in test_dl])
            test_loss[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            correct = 0
            total = 0
            for data in test_dl:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[epoch]}, Test Loss {test_loss[epoch]}, Accuracy: {100*correct/total}')
    return train_loss, test_loss


if __name__ == "__main__":
    # Define Hyperparameters
    lr = 0.001
    # Load Data
    train_data = CamVidDataset(image_path='./CamVid/train', label_path='./CamVid/train_labels',transform=transform)
    test_data = CamVidDataset(image_path='./CamVid/val', label_path='./CamVid/vak_labels',transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)
    # Instantiate Model
    model = SegNet()
    #Define Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train
    train(model, trainloader, testloader, optimizer, criterion)
    # Save Model to file
    with f = open('segnet.pkl', 'wb'):
        pickle.dump(model,f)
        f.close()
