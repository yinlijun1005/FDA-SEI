import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn_onlycnn import *
from get_dataset_sourcetrain import *
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(model, loss, train_dataloader, optimizer, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    classifier_loss = 0
    result_loss = 0

    for data_nnl in train_dataloader:
        data, target = data_nnl
        target = target.long().squeeze()

        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        classifier_output = F.log_softmax(output[1], dim=1)
        classifier_loss_batch = loss(classifier_output, target)
        result_loss_batch = classifier_loss_batch
        result_loss_batch.backward()
        optimizer.step()
        classifier_loss += classifier_loss_batch.item()
        result_loss += result_loss_batch.item()

        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    classifier_loss /= len(train_dataloader)
    result_loss /= len(train_dataloader)

    print('Train Epoch: {} \tClassifier_Loss: {:.6f}, Combined_Loss, :{:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        classifier_loss,
        result_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('ClassifierLoss/train', classifier_loss, epoch)

def test(model, loss, test_dataloader, epoch, writer, device_num):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long().squeeze()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output[1], dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('ClassifierLoss/validation', test_loss, epoch)

    return test_loss

def train_and_test(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer,save_path, device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch, writer, device_num)
        test_loss = test(model, loss_function, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        #     writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

def evaluate(model, test_dataloader, device_num):
    model.eval()
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long().squeeze()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

# def Data_prepared(num):
#     X_train, X_val, Y_train, Y_val = TrainDataset(num)
#
#     min_value = X_train.min()
#     min_in_val = X_val.min()
#     if min_in_val < min_value:
#         min_value = min_in_val
#
#     max_value = X_train.max()
#     max_in_val = X_val.max()
#     if max_in_val > max_value:
#         max_value = max_in_val
#
#     return max_value, min_value

def TrainDataset_prepared(num, snr):
    X_train, X_val, Y_train, Y_val = TrainDataset(num, snr)

    # max_value, min_value = Data_prepared(num)
    #
    # X_train = (X_train - min_value) / (max_value - min_value)
    # X_val = (X_val - min_value) / (max_value - min_value)

    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)


    return X_train, X_val, Y_train, Y_val

def TestDataset_prepared(snr):
    X_test, Y_test = TestDataset(snr)

    # max_value, min_value = Data_prepared(n_classes)
    #
    # X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test

class Config:
    def __init__(
        self,
        batch_size: int = 64,
        test_batch_size: int = 64,
        epochs: int = 100,
        lr: float = 0.001,
        n_classes: int = 6,

        device_num: int = 0,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_classes = n_classes

        self.device_num = device_num

def main(RANDOM_SEED, snr):
    writer = SummaryWriter("logs_16psk_TL_bpsk_SourceModelTrain")
    device = torch.device("cuda:"+str(conf.device_num))

    set_seed(RANDOM_SEED)

    X_train, X_val, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes, snr)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)


    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

    X_test, Y_test = TestDataset_prepared(snr)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True)


    model = base_complex_model()
    save_path = 'model_weight_Source_16psk/16psk2000_sourceModel_snr=' + str(snr) + 'dB.pth'

    if torch.cuda.is_available():
        model = model.to(device)
    print(model)

    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)

    train_and_test(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optim, epochs=conf.epochs, writer=writer, save_path=save_path, device_num=conf.device_num)

    # test

    model = torch.load(save_path)
    print(model)
    acc = evaluate(model, test_dataloader, conf.device_num)
    return acc

    #test
    # model = torch.load(conf.save_path)
    # print(model)
    # X_test, Y_test = TestDataset_prepared(conf.n_classes)
    # test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # evaluate(model, test_dataloader, conf.device_num)
if __name__ == '__main__':
    conf = Config()
    for snr in range(0, 12, 2):
        acc = main(2023, snr)
        print(acc)

    # for snr in range(0, 12, 2):
    #     X_test, Y_test = TestDataset_prepared(snr)
    #     test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    #     test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True)
    #
    #     save_path = 'model_weight_Source_16psk/16psk2000_sourceModel_snr=' + str(snr) + 'dB.pth'
    #     model = torch.load(save_path)
    #     # print(model)
    #     acc = evaluate(model, test_dataloader, conf.device_num)
    #     print(acc)
    #


# if __name__ == '__main__':
#     conf = Config()
#     for snr in range(0, 12, 2):
#         acc_all = np.zeros((5, 1))
#         for i in range(0, 5):
#             acc_all[i] = main(2023+i, snr)
#         print(acc_all)
#         data_Y_pred = pd.DataFrame(acc_all)
#         writer = pd.ExcelWriter("model_accuracy/16psk2000_SourceAccuracy_" + str(snr)+ "dB.xlsx")
#         data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
#         writer.save()
#         writer.close()



