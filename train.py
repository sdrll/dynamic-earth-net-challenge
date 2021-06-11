# PyTorch
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from siamunet_diff import SiamUnet_diff
from utils import ChangeDetectionDataset

semi_supervised_labels_path = '../dataset/semisupervised_labels/Labels'
save_model_path = 'weights/'
net, net_name = SiamUnet_diff(4, 8), 'FC-Siam-diff'
optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
n_epochs = 10
batch_size = 40
n_classes = 8
writer = SummaryWriter()


def train(train_loader):
    net.cuda()

    t = np.linspace(1, n_epochs, n_epochs)

    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        class_accuracy = list(0. for i in range(n_classes))
        tot_loss = 0
        tot_count = 0
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(n_epochs))

        for batch in train_loader:
            I1 = batch['I1'].float().cuda()
            I2 = batch['I2'].float().cuda()
            label = batch['label'].cuda()
            optimizer.zero_grad()
            output = net(I1, I2)
            loss = loss_fn(output, label.long())
            loss.backward()
            optimizer.step()
            tot_loss += loss.data * np.prod(label.size())
            tot_count += np.prod(label.size())
            _, predicted = torch.max(output.data, 1)
            cnf_matrix = confusion_matrix(label.data.int().cpu().numpy().flatten(),
                                          predicted.int().cpu().numpy().flatten(), labels=[0, 1, 2, 3, 4, 5, 6, 7])
            fp += cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            fn += cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tp += np.diag(cnf_matrix)
            print(cnf_matrix)
            tn += cnf_matrix.sum() - (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) + cnf_matrix.sum(axis=1) - np.diag(
                cnf_matrix) + np.diag(cnf_matrix))

            # c = (predicted.int() == label.data.int())
            # for i in range(c.size(1)):
            #     for j in range(c.size(2)):
            #         l = int(label.data[0, i, j])
            #         class_correct[l] += c[0, i, j]
            #         class_total[l] += 1
            # print(len(class_correct))
            # print(len(class_total))
            # print(c)
            # pr = (predicted.int() > 0).cpu().numpy()
            # gt = (label.data.int() > 0).cpu().numpy()
            # tp += np.logical_and(pr, gt).sum()
            # tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
            # fp += np.logical_and(pr, np.logical_not(gt)).sum()
            # fn += np.logical_and(np.logical_not(pr), gt).sum()
        scheduler.step()

        net_loss = tot_loss / tot_count
        net_accuracy = 100 * (np.frombuffer(tp).sum() + np.frombuffer(tn).sum()) / tot_count
        prec = np.frombuffer(tp).sum() / (np.frombuffer(tp).sum() + np.frombuffer(fp).sum())
        rec = np.frombuffer(tp).sum() / (np.frombuffer(tp).sum() + np.frombuffer(fn).sum())
        writer.add_scalar("Loss/train", net_loss, epoch_index)
        writer.add_scalar("Accuracy/train", net_accuracy, epoch_index)
        writer.add_scalar("Precision/train", prec, epoch_index)
        writer.add_scalar("Recall/train", rec, epoch_index)


train_dataset = ChangeDetectionDataset(semi_supervised_labels_path, transform=None)
weights = torch.FloatTensor(train_dataset.weights).cuda()
loss_fn = nn.NLLLoss(weight=weights)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

t_start = time.time()
train(train_loader)
t_end = time.time()
print('Elapsed time: {}'.format(t_end - t_start))
writer.flush()
writer.close()
torch.save(net.state_dict(), save_model_path + '18-05-10epochs_INS')
