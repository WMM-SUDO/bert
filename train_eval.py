# coding: UTF-8
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        train_loss_all = []
        val_loss_all = []
        for i, (trains, labels) in enumerate(train_iter):
            # labels [1,2,3]有几个标签就是几个
            outputs = model(trains)
            labels = labels.float()
            model.zero_grad()
            # loss = F.cross_entropy(outputs, labels)
            m = nn.Sigmoid()
            loss_func = nn.BCELoss()
            out_sigm = m(outputs)
            loss = loss_func(out_sigm, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0 or (total_batch + 2) % len(train_iter) == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # pred_list = torch.where(out_sigm.cpu() < 0.3, 0, 1).data.numpy().tolist()  # todo
                pred_list = torch.where(out_sigm.cpu() < 0.5, torch.zeros_like(out_sigm).cpu(), torch.ones_like(out_sigm).cpu()).data.numpy().tolist()

                train_acc = cal_acc(np.array(pred_list), np.array(labels.data.cpu().numpy().tolist()))

                # true = labels.data.cpu()
                # predic = torch.max(outputs.data, 1)[1].cpu()
                # train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    # 如果当前模型比较好，loss比之前的最佳loss低，就更新best_loss, 然后save，打个*
                    # 没有✳就表示没有更好，可以不train了？
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)

                np.set_printoptions(threshold=np.inf)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                # label_index = []
                # for la in labels:
                #     l.append(np.array(la.data.cpu().numpy().tolist()).nonzero()[0])
                #     print(config.class_list[np.array(la.data.numpy().cpu().int().tolist()).nonzero()[0]])
                model.train()

            if (total_batch + 2) % len(train_iter) == 0:
                # print(dev_loss.data.cpu(),"---")
                # 每多少轮输出在训练集和验证集上的效果
                # pred_list = torch.where(out_sigm.cpu() < 0.3, 0, 1).data.numpy().tolist()
                #这个在服务器不能运行，where的第二、三个参数也必须是tensor
                pred_list = torch.where(out_sigm.cpu() < 0.5, torch.zeros_like(out_sigm).cpu(), torch.ones_like(out_sigm).cpu()).data.numpy().tolist()
                train_acc = cal_acc(np.array(pred_list), np.array(labels.data.cpu().numpy().tolist()))

                # true = labels.data.cpu()
                # predic = torch.max(outputs.data, 1)[1].cpu()
                # train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    # 如果当前模型比较好，loss比之前的最佳loss低，就更新best_loss, 然后save，打个*
                    # 没有✳就表示没有更好，可以不train了？
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)

                train_loss_all.append(loss.item())
                val_loss_all.append(dev_loss.data.cpu())

                np.set_printoptions(threshold=np.inf)
                msg = '\nIter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}\n'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print("=========================================prediction on train data:")
                print(np.array(pred_list)[:10].nonzero()[0])
                print(np.array(pred_list)[:10].nonzero()[1])
                print("=========================================train data labels:")
                print(np.array(labels.data.cpu().numpy().tolist())[:10].nonzero()[0])
                print(np.array(labels.data.cpu().numpy().tolist())[:10].nonzero()[1])
                print("-----------------------------------------prediction on train data:")
                print(np.round(np.array(out_sigm.cpu().data.numpy())[:10], 4))
                print("-----------------------------------------train data labels:")
                print(np.array(labels.data.cpu().numpy().tolist())[:10])
                label_index = []
                # for la in labels:
                #     l.append(np.array(la.data.cpu().numpy().tolist()).nonzero()[0])
                #     print(config.class_list[np.array(la.data.numpy().cpu().int().tolist()).nonzero()[0]])
                model.train()

            total_batch += 1


            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
            #to do
        if flag:
            break
        print("Train loss all: ", train_loss_all)
        print("\nVal loss all: ", val_loss_all)

    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter, test=False)
    # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    # print("Precision, Recall and F1-Score...")
    # print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def cal_acc(pred, label):
    correct = pred - label
    # counts = label.shape[0]*label.shape[1]
    counts = sum(sum(label != 0))
    wrong = sum(sum(correct == -1))
    acc = 1.0 * (counts - wrong) / counts
    return acc


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0

    pred_all_list = []
    labels_all_lsit = []

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            labels = labels.float()
            outputs = model(texts)

            # loss = F.cross_entropy(outputs, labels)
            m = nn.Sigmoid()
            loss_func = nn.BCELoss()
            out_sigm = m(outputs)
            loss = loss_func(out_sigm, labels)

            loss_total += loss

            # pred_list = torch.where(outputs.cpu() < 0.5, 0, 1).data.numpy().tolist()  # todo
            pred_list = torch.where(out_sigm.cpu() < 0.5, torch.zeros_like(out_sigm).cpu(),
                                    torch.ones_like(out_sigm).cpu()).data.numpy().tolist()

            # [batch_size,label_size] 小于这个阈值的被设为0
            pred_all_list.extend(pred_list)

            labels = labels.data.cpu().numpy()

            labels_all_lsit.extend(labels.tolist())
            # print("----", len(labels_all_lsit), "-----", len(pred_all_list))
            # predic_prob = torch.max(outputs.data, 1)[0].cpu().numpy()
            # predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # predic_12 = []
            # #一个batch里面，每一个预测的概率大于1.2的标签，[batch_size, N] N的数目不确定
            # for out in outputs:
            #     # print(out.data[out.data > 1.2])
            #     out_np = out.cpu().numpy()
            #
            #     predic_12 = np.append(predic_12, np.where(out_np > 1.2))
            # print(predic_12)
            # print('=====================')
            #
            # label_index = np.array([], dtype=int)
            # for label in labels:
            #     label_index = np.append(label_index, label.nonzero()[0][0])
            #     print(label.nonzero())
            #
            # labels_all = np.append(labels_all, label_index)  # todo
            # predict_all = np.append(predict_all, predic)
            # print("+++",label_index.shape, "----------", predic.shape)

    # print(labels_all.shape, "----------", predict_all.shape)
    acc = cal_acc(np.array(pred_all_list), np.array(labels_all_lsit))
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
