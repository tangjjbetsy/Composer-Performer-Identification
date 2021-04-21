import os
import time
import glob
import torch
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import network
from config import *



class DealDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_loader(data_X, data_y):    
    data = DealDataset(data_X, data_y)
    size = data.len
    loader = DataLoader(dataset=data,           
                    batch_size=BATCH_SIZE, 
                    shuffle=shuffle,
                    num_workers=num_workers)
    return loader

def checkpoint(net, save_path, acc, loss, iterations):
    snapshot_prefix = os.path.join(save_path, 'snapshot_' + net._class_name())
    snapshot_path = snapshot_prefix + '_acc_{:.2f}_loss_{:.4f}_iter_{}_model.pt'.format(acc, loss.item(), iterations)
    torch.save(net, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

def train(optimizer, criterion, net, device, epoches, save_path=SAVEPATH):
    iterations = 0
    start = time.time()
    
    best_dev_acc = -1; best_snapshot_path = ''
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f},{:8.4f},{:12.4f},{:12.4f}'.split(','))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f},{},{:12.4f},{}'.split(','))
    
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    print(header)

    train_loader = data_loader(np.load("X_train.npy"), np.load("y_train.npy"))
    dev_loader = data_loader(np.load("X_val.npy"), np.load("y_val.npy"))

    for epoch in range(epoches):  # loop over the dataset multiple times
        correct, total = 0, 0
        for i, data in enumerate(train_loader, 0):
            iterations += 1
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float()) # torch.Size([32, 6])
            labels = labels.view(-1) - 1  # torch.Size([32, 1])
            
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()  
        
            # compute accuracy 
            acc = correct / total * 100
        
            # checkpoint model periodically
            if iterations % SAVE_EVERY == 0:
                checkpoint(net, save_path, acc, loss, iterations)
            
            # validation model periodically
            if iterations % DEV_EVERY == 0:
                # calculate accuracy on validation set
                dev_correct, dev_total = 0, 0
                with torch.no_grad():
                    for dev_batch_idx, dev_batch in enumerate(dev_loader, 0):
                        signals, labels = dev_batch
                        signals = signals.to(device)
                        labels = labels.to(device)
                        labels = labels.view(-1) - 1
                        
                        predicts = net(signals.float())
                        dev_loss = criterion(predicts, labels)
                        dev_correct += (torch.max(predicts, 1)[1].view(-1) == labels).sum().item()
                        dev_total += labels.size(0)
                dev_acc = 100. * dev_correct / dev_total

                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(train_loader),
                    100. * (1+i) / len(train_loader), loss.item(), dev_loss.item(), acc, dev_acc))

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:

                    # found a model with better validation set accuracy

                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(save_path, 'best_snapshot_' + net._class_name())
                    best_snapshot_path = snapshot_prefix + '_devacc_{:.2f}_devloss_{:.4f}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(net, best_snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != best_snapshot_path:
                            os.remove(f)

            elif iterations % LOG_EVERY == 0:
                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(train_loader),
                    100. * (1+i) / len(train_loader), loss.item(), ' '*8, acc, ' '*12))

    print('Finished Training')
    return net, best_snapshot_path

def test(net, fp, validation, device):
    correct = 0
    total = 0
    
    test_loader = data_loader("test", True)
    y_pred = []
    y_true = []

    if validation:
        net = torch.load(fp)

    with torch.no_grad():
        for data in test_loader:
            signals, labels = data
            signals = signals.to(device)
            labels = labels.to(device)
            for i in labels.view(-1):
                y_true.append(i.view(-1).tolist())
            
            outputs = net(signals.float())
            predicted = outputs.data.argmax(dim=1)
            for i in predicted.view(-1) + 1:
                y_pred.append(i.view(-1).tolist())
            
            labels = labels.view(-1) - 1
            correct += (predicted.view(-1) == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(total,correct)
    print('Accuracy: %.2f %%' % acc)
    return acc, np.ravel(y_true), np.ravel(y_pred)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        device = torch.device('cuda:{}'.format(1))
        print("Using GPU for training")
    else:
        device = torch.device('cpu')

    print('\n----------------------------- EXPERIMENT -----------------------------')
    net = network.resnet50().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    net, fp = train(optimizer, criterion, net, device, 50)