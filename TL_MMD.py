import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn_onlycnn import *
from get_dataset_transferlearning import *
import random
import os
import pandas as pd
from MMD_Loss import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def train_withMMD(model_target, model_source, loss_nll, loss_mmd, train_dataloader_target, train_dataloader_source, learning_rate, batchsize, weight_mmd, epoch, writer, device_num):
    model_target.train()
    model_source.eval()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    classifier_loss_target = 0
    combine_loss = 0
    mmd_loss = 0
    optimizer = torch.optim.Adam(model_target.parameters(), lr=learning_rate)

    for i, data in enumerate(zip(train_dataloader_target, train_dataloader_source)):
        data_t = data[0][0]
        target_t = data[0][1]
        data_s = data[1][0]
        target_s = data[1][1]
        target_t = target_t.squeeze().long()
        target_s = target_s.squeeze().long()

        if torch.cuda.is_available():
            data_t = data_t.to(device)
            target_t = target_t.to(device)
            data_s = data_s.to(device)
            target_s = target_s.to(device)

        optimizer.zero_grad()

        output_target = model_target(data_t)
        output_source = model_source(data_s)

        classifier_output_target = F.log_softmax(output_target[1], dim=1)
        classifier_loss_batch_target = loss_nll(classifier_output_target, target_t)  #log_softmax + NLLLoss = CrossEntropyLoss
        mmd_loss_batch = loss_mmd(output_source[0], output_target[0])
        combine_loss_batch = classifier_loss_batch_target + weight_mmd * mmd_loss_batch
        combine_loss_batch.backward()
        optimizer.step()

        classifier_loss_target += classifier_loss_batch_target.item()
        mmd_loss += mmd_loss_batch.item()
        combine_loss += combine_loss_batch.item()

        pred = classifier_output_target.argmax(dim=1, keepdim=True)     #(batch_size, num_classes)
        correct += pred.eq(target_t.view_as(pred)).sum().item()

    number_dataloader = i+1
    classifier_loss_target /= number_dataloader
    mmd_loss /= number_dataloader
    combine_loss /= number_dataloader
    number_training_samples = number_dataloader * batchsize
    print('Train Epoch: {} \tClassifier_Loss: {:.6f}, MMD_Loss: {:.6f}, Combine_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        classifier_loss_target,
        mmd_loss,
        combine_loss,
        correct,
        number_training_samples,
        100.0 * correct / number_training_samples)
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / number_training_samples, epoch)
    writer.add_scalar('ClassifierLoss/train', classifier_loss_target, epoch)
    writer.add_scalar('MMDLoss/train', mmd_loss, epoch)

def test_withMMD(model_target, model_source, loss_nll, loss_mmd, test_dataloader_target, test_dataloader_source, batchsize, weight_mmd, epoch, writer, device_num):
    model_target.eval()
    model_source.eval()

    nll_loss_value = 0
    mmd_loss_value = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))

    with torch.no_grad():
        for i, data in enumerate(zip(test_dataloader_target, test_dataloader_source)):
            data_t = data[0][0]
            target_t = data[0][1]
            data_s = data[1][0]
            target_s = data[1][1]
            target_t = target_t.squeeze().long()
            target_s = target_s.squeeze().long()

            if torch.cuda.is_available():
                data_t = data_t.to(device)
                target_t = target_t.to(device)
                data_s = data_s.to(device)
                target_s = target_s.to(device)

            output_target = model_target(data_t)
            output_source = model_source(data_s)

            output_t = F.log_softmax(output_target[1], dim=1)
            nll_loss_value += loss_nll(output_t, target_t).item()
            mmd_loss_value += loss_mmd(output_source[0], output_target[0]).item()

            pred = output_t.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_t.view_as(pred)).sum().item()

        number_dataloader = i + 1
        number_training_samples = number_dataloader * batchsize

        nll_loss_value /= number_dataloader
        mmd_loss_value /= number_dataloader
        test_loss = nll_loss_value + weight_mmd * mmd_loss_value
        fmt = '\nValidation set: Classifier_Loss: {:.6f}, MMD_Loss: {:.6f}, Combined_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'
        print(
            fmt.format(
            nll_loss_value,
            mmd_loss_value,
            test_loss,
            correct,
            number_training_samples,
            100.0 * correct / number_training_samples,)
        )

    writer.add_scalar('Accuracy/validation', 100.0 * correct / number_training_samples, epoch)
    writer.add_scalar('ClassifierLoss/validation', nll_loss_value, epoch)
    writer.add_scalar('MMDLoss/validation', mmd_loss_value, epoch)

    return test_loss

def train_and_test(model_target,
                   model_source,
                   nll_loss_function,
                   mmd_loss_function,
                   train_dataset_target,
                   train_dataloader_source,
                   val_dataset_target,
                   val_dataloader_source,
                   learning_rate,
                   batch_size,
                   weight_mmd,
                   epochs,
                   writer,
                   save_targetmodel,
                   device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):

        train_dataloader_target = DataLoader(train_dataset_target, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader_target = DataLoader(val_dataset_target, batch_size=batch_size, shuffle=True, drop_last=True)

        train_withMMD(model_target,
                      model_source,
                      nll_loss_function,
                      mmd_loss_function,
                      train_dataloader_target,
                      train_dataloader_source,
                      learning_rate,
                      batch_size,
                      weight_mmd,
                      epoch,
                      writer,
                      device_num)

        test_loss = test_withMMD(model_target,
                         model_source,
                         nll_loss_function,
                         mmd_loss_function,
                         val_dataloader_target,
                         val_dataloader_source,
                         batch_size,
                         weight_mmd,
                         epoch,
                         writer,
                         device_num)

        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model_target, save_targetmodel)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def evaluate(model, test_dataloader, device_num):
    model.eval()
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # target_pred[len(target_pred):len(target)-1] = pred.tolist()
            # target_real[len(target_real):len(target)-1] = target.tolist()

    #     target_pred = np.array(target_pred)
    #     target_real = np.array(target_real)
    #
    # # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("CVNN_ADS_B_0_1023/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()
    #
    # # 将原始标签存下来
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("CVNN_ADS_B_0_1023/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return 100.0 * correct / len(test_dataloader.dataset)

# def SourceData_prepared(snr):
#     X_train, X_val, value_Y_train, value_Y_val = Source_TrainDataset(snr)
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

def SourceTrainDataset_prepared(snr):
    X_train, X_val, value_Y_train, value_Y_val = Source_TrainDataset(snr)

    # max_value, min_value = SourceData_prepared(snr)
    #
    # X_train = (X_train - min_value) / (max_value - min_value)
    # X_val = (X_val - min_value) / (max_value - min_value)

    return  X_train, X_val, value_Y_train, value_Y_val

def TargetTrainDataset_prepared(snr, shot):
    X_train, X_val, value_Y_train, value_Y_val = Target_TrainDataset(snr, shot)
    X_test, value_Y_test = TestDataset(snr)

    # min_value = X_train.min()
    # min_in_val = X_val.min()
    # if min_in_val < min_value:
    #     min_value = min_in_val
    #
    # max_value = X_train.max()
    # max_in_val = X_val.max()
    # if max_in_val > max_value:
    #     max_value = max_in_val
    #
    # X_train = (X_train - min_value) / (max_value - min_value)
    # X_val = (X_val - min_value) / (max_value - min_value)
    # X_test = (X_test - min_value) / (max_value - min_value)

    return  X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test

class Config:
    def __init__(
        self,
        batch_size: int = 50,
        test_batch_size: int = 20,
        epochs: int = 100,
        lr: float = 0.001,
        n_classes_source: int = 6,
        n_classes_target: int = 6,
        shot: int = 30,
        weight_mmd: float = 2.5,
        device_num: int = 0,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_mmd = weight_mmd
        self.n_classes_source = n_classes_source
        self.n_classes_target = n_classes_target
        self.shot = shot
        self.device_num = device_num

def main(RANDOM_SEED, snr):

    writer = SummaryWriter("logs/logs_Frozen6_and_targetlossMMD" + str(conf.shot) + "shot_" + str(snr) + "dB")
    device = torch.device("cuda:"+str(conf.device_num))

    set_seed(RANDOM_SEED)

    #source data: ADS-B
    X_train_source, X_val_source, value_Y_train_source, value_Y_val_source = SourceTrainDataset_prepared(snr)

    train_dataset_source = TensorDataset(torch.Tensor(X_train_source), torch.Tensor(value_Y_train_source))
    train_dataloader_source = DataLoader(train_dataset_source, batch_size=conf.batch_size, shuffle=True, drop_last=True)

    val_dataset_source = TensorDataset(torch.Tensor(X_val_source), torch.Tensor(value_Y_val_source))
    val_dataloader_source = DataLoader(val_dataset_source, batch_size=conf.batch_size, shuffle=True, drop_last=True)

    #target data: ADS-B
    X_train_target, X_val_target, X_test, Y_train_target, Y_val_target, Y_test = TargetTrainDataset_prepared(snr, conf.shot)
    train_dataset_target = TensorDataset(torch.Tensor(X_train_target), torch.Tensor(Y_train_target))
    val_dataset_target = TensorDataset(torch.Tensor(X_val_target), torch.Tensor(Y_val_target))

    nll_loss = nn.NLLLoss()
    mmd_loss = MMD_loss()
    if torch.cuda.is_available():
        nll_loss = nll_loss.to(device)
        mmd_loss = mmd_loss.to(device)

    save_sourcemodel = 'Source/model_weight_Source_16psk/16psk2000_sourceModel_snr=' + str(snr) + 'dB.pth'
    save_targetmodel = 'model_weight/16psk_bpsk/16psk_bpsk_Frozen6_and_targetlossMMD' + str(conf.shot) + 'shot_snr=' + str(snr) + 'dB.pth'

    model_source = torch.load(save_sourcemodel)
    model_target = torch.load(save_sourcemodel)
    model_target.base_complex_model2.linear2 = nn.LazyLinear(conf.n_classes_target)
    for para in model_target.base_complex_model1.parameters():                  ################
        para.requires_grad = False
    for para in model_target.base_complex_model1.batchnorm7.parameters():
        para.requires_grad = True
    for para in model_target.base_complex_model1.conv7.parameters():
        para.requires_grad = True
    if torch.cuda.is_available():
        model_source = model_source.to(device)
        model_target = model_target.to(device)

    print(model_source)
    print(model_target)

    train_and_test(model_target=model_target,
                   model_source=model_source,
                   nll_loss_function=nll_loss,
                   mmd_loss_function=mmd_loss,
                   train_dataset_target=train_dataset_target,
                   train_dataloader_source=train_dataloader_source,
                   val_dataset_target=val_dataset_target,
                   val_dataloader_source=val_dataloader_source,
                   learning_rate=conf.lr,
                   batch_size=conf.batch_size,
                   weight_mmd=conf.weight_mmd,
                   epochs=conf.epochs,
                   writer=writer,
                   save_targetmodel=save_targetmodel,
                   device_num=conf.device_num)

    #test
    model = torch.load(save_targetmodel)
    print(model)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    acc = evaluate(model, test_dataloader, conf.device_num)
    return acc

if __name__ == '__main__':
    conf = Config()
    for snr in range(0, 12, 2):
        acc_all = np.zeros((20,1))
        for i in range(0, 20):
            acc_all[i] = main(2023+i, snr)
        print(acc_all)
        data_Y_pred = pd.DataFrame(acc_all)
        writer = pd.ExcelWriter("accuracy/16psk_bpsk_Frozen6_and_targetlossMMD"+str(conf.shot)+"shot_" + str(snr)+ "dB_20monte.xlsx")
        data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()
