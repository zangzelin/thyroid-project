import os
from argparse import ArgumentParser
from functools import partial
from operator import mod
from random import sample

import numpy as np
# from main import main
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import wandb
from analysis import Analysis, ROCModelCompair
from loaddata import (Fill_nan, Get_protein_index, Get_selected_protein_matrix,
                      Loaddata, LoaddataAC, LoaddataPRO34, StanFeature)
from model import MLP, TPDNet
from tool import SetSeed

# from wandb import init as wandbinit
# from wandb import log as wandblog
torch.set_num_threads(1)


def Param():
    parser = ArgumentParser(description="zelin zang author")
    parser.add_argument("--name", type=str, default="TPD")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed_feature", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--fillna", type=float, default=12.0)
    parser.add_argument("--stan_feature", type=int, default=0)
    # parser.add_argument("--protein_list_input", type=float, default=['P04792', 'O14964', 'P04899', 'P78527', 'P00568', 'P42224', 'O00339', 'P02751', 'P17931', 'O75347', 'Q9HAT2', 'P04216', 'P02765', 'P35579', 'P30086', 'P10909', 'P16403', 'P27797', 'P04083', 'P26038'])
    parser.add_argument("--project", type=str, default='TPD2021_v5')
    # parser.add_argument("--protein_list_input", type=float, default=['P04792', 'O14964', 'P04899', 'P78527', 'P00568', 'P42224', 'O00339', 'P02751', 'P17931', 'O75347', 'Q9HAT2', 'P04216', 'P02765', 'P35579', 'P30086', 'P10909', 'P16403', 'P27797', 'P04083', 'P26038'])
    # parser.add_argument("--protein_list_input", type=float, default=['O75347', 'O00339', 'P00568', 'P08581', 'Q07812', 'O14964', 'P78527', 'P27797', 'P04792', 'P35579', 'P04899', 'P26038', 'P02765', 'P17931', 'Q9HAT2', 'P04216', 'P10909', 'P30086', 'P42224', 'P16403'])
    parser.add_argument("--protein_list_input", nargs='+', default=['P02765', 'P04083', 'O00339', 'P58546', 'O75347', 'P04216', 'P02751',
                        'P83731', 'P00568', 'P78527', 'P04792', 'P57737', 'P42224', 'P27797', 'Q9HAT2', 'P30086', 'O14964', 'P10909', 'P17931'])
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args().__dict__
    return args


def Train_network(data, label, args):
    # train the network with 5-fold cross validation
    # inputs:
    #     data: the input data for train and validation 
    #     label: the ground-truth label for train and validation 
    #     args: detail settings
    # outputs:
    #     auc_list: the validation auc list for every single fold
    #     model_list: the model list for every single fold

    N = data.shape[0]
    index_shuffle = np.array(sample(range(0, N), N))
    data = data[index_shuffle]
    label = label[index_shuffle]

    if args['device'] == 'cuda':
        data = torch.tensor(data).to('cuda') 
        label = torch.tensor(label).to('cuda')

    auc_list = []
    model_list = []

    kfTool = KFold(n_splits=5, random_state=args['seed'], shuffle=True)
    for i, (trainIndex, valIndex) in enumerate(kfTool.split(data)):
        XTrain, Xval = data[trainIndex], data[valIndex]
        yTrain, yval = label[trainIndex], label[valIndex]
        net = TPDNet(
            partial(MLP, inputShape=data.shape[1]),
            optimizer=torch.optim.AdamW,
            criterion=partial(torch.nn.CrossEntropyLoss,
                              weight=torch.tensor([args["alpha"], 2 - args["alpha"]])),
            max_epochs=args["epochs"],
            lr=args["lr"],
            device=args["device"],  # uncomment this to train with CUDA
            verbose=args["verbose"],
            optimizer__weight_decay=args["l2"],
            train_split=None,
            batch_size=args["batch_size"],
        )
        # net.initialize()
        XTrain = XTrain.float()
        yTrain = yTrain.long()
        net.fit(X=XTrain, y=yTrain)

        Xval = Xval.float()
        yval = yval.long().detach().cpu().numpy()
        y_pre = net.predict_proba(Xval)[:, 1]
        auc = roc_auc_score(yval, y_pre)

        auc_list.append(auc)
        model_list.append(net)
    return auc_list, model_list


def Test_network(net, data, label):
    # test the network by auc and acc
    # inputs:
    #     net: the model that need to be test
    #     data: the input data for testing
    #     label: the ground-truth label for testing 
    # outputs:
    #     auc: the auc of the test results
    #     acc: the acc of the test results

    Xval = torch.tensor(data).float()
    yval = torch.tensor(label).long()
    y_pre = net.predict_proba(Xval)[:, 1]

    auc = roc_auc_score(yval, y_pre)
    y_pre_bool = np.copy(y_pre)
    y_pre_bool[y_pre_bool < 0.5] = 0
    y_pre_bool[y_pre_bool >= 0.5] = 1
    acc = accuracy_score(yval, y_pre_bool)

    return auc, acc


def Find_threshold(net, data, label):
    # find the best threshold with the validation set
    # inputs:
    #     net: the model that need to be test
    #     data: the input data for testing
    #     label: the ground-truth label for testing 
    # outputs:
    #     threshold_list[best_index]: the best threshold for binary classification


    Xval = torch.tensor(data).float()
    yval = torch.tensor(label).long()
    y_pre = net.predict_proba(Xval)[:, 1]

    # auc = roc_auc_score(yval, y_pre)

    threshold_list = [i/500 for i in range(500)]
    acc_list = []
    for threshold in threshold_list:
        y_pre_bool = np.copy(y_pre)
        y_pre_bool[y_pre_bool < threshold] = 0
        y_pre_bool[y_pre_bool >= threshold] = 1
        acc = accuracy_score(yval, y_pre_bool)
        acc_list.append(acc)

    best_index = acc_list.index(max(acc_list))
    return threshold_list[best_index]


def main():
    # the main function of the code

    # load the setting and set the random seed
    args = Param()
    protein_list = args['protein_list_input']
    # protein_list = sample(protein_list, len(protein_list))
    # print(protein_list)
    args['protein_list'] = protein_list
    SetSeed(args["seed"])

    # use wandb as loggger, see: https://wandb.ai/site
    if args['use_wandb'] == 1:
        wandb.init(
            name='_'.join([args["name"], str(args["lr"]), str(args["l2"]),
                           str(args["epochs"]), str(args["alpha"])]),
            project=args["project"],
            entity='zangzelin',
            mode='online',
            save_code=True,
            config=args,
        )

    # load the dataset for training, validation and testing
    patient_dis, label_dis, protein_dis, patient_ret, label_ret, protein_ret, patient_pro, label_pro, protein_pro, protein_name = Loaddata(
        'data/alldata.csv')
    patient_AC, label_AC, protein_AC = LoaddataAC(
        'data/alldata_Histopathology_type_list.csv')
    patient_dis_new, label_dis_new, protein_dis_new, patient_ret_new, label_ret_new, protein_ret_new, patient_pro_new, label_pro_new, protein_pro_new, protein_name_new = Loaddata(
        'data_Jul1/alldata_Histopathology_type_list.csv')
    patient_AC_new, label_AC_new, protein_AC_new = LoaddataAC(
        'data_Jul1/alldata_Histopathology_type_list.csv')
    patient_PRO34_new, label_PRO34_new, protein_PRO34_new = LoaddataPRO34(
        'data_Jul1/alldata_Histopathology_type_list0704.csv')
    index_list = Get_protein_index(protein_name, protein_list)

    # select the feature
    data_dis = Get_selected_protein_matrix(protein_dis, index_list)
    data_ret = Get_selected_protein_matrix(protein_ret, index_list)
    data_pro = Get_selected_protein_matrix(protein_pro, index_list)
    data_pro_new = Get_selected_protein_matrix(
        protein_pro_new, index_list).astype(float)
    data_AC = Get_selected_protein_matrix(protein_AC, index_list)
    data_AC_new = Get_selected_protein_matrix(
        protein_AC_new, index_list).astype(float)
    data_PRO34_new = Get_selected_protein_matrix(
        protein_PRO34_new, index_list).astype(float)

    # fill the nan value
    data_dis = Fill_nan(data_dis, args['fillna'])
    data_ret = Fill_nan(data_ret, args['fillna'])
    data_pro = Fill_nan(data_pro, args['fillna'])
    data_pro_new = Fill_nan(data_pro_new, args['fillna'])
    data_AC = Fill_nan(data_AC, args['fillna'])
    data_AC_new = Fill_nan(data_AC_new, args['fillna'])
    data_PRO34_new = Fill_nan(data_PRO34_new, args['fillna'])

    if args['stan_feature']:
        data_dis = StanFeature(data_dis)
        data_ret = StanFeature(data_ret)
        data_pro = StanFeature(data_pro)
        data_pro_new = StanFeature(data_pro_new)
        data_AC = StanFeature(data_AC)
        data_AC_new = StanFeature(data_AC_new)
        data_PRO34_new = StanFeature(data_PRO34_new)

    # normalize the data
    normalizer = StandardScaler()
    normalizer.fit(data_dis)
    data_dis = normalizer.transform(data_dis)
    data_ret = normalizer.transform(data_ret)
    data_pro = normalizer.transform(data_pro)
    data_pro_new = normalizer.transform(data_pro_new)
    data_AC = normalizer.transform(data_AC)
    data_AC_new = normalizer.transform(data_AC_new)
    data_PRO34_new = normalizer.transform(data_PRO34_new)

    # train the network, and select the best one with validation auc 
    auc_list, model_list = Train_network(data_dis, np.array(label_dis), args)
    max_index = auc_list.index(max(auc_list))

    # test the network 
    # print('model', max_index, 'train auc', auc_list[max_index])
    ret_auc, ret_acc = Test_network(
        model_list[max_index], data_ret, np.array(label_ret))
    pro_auc, pro_acc = Test_network(
        model_list[max_index], data_pro, np.array(label_pro))
    pro_auc_new, pro_acc_new = Test_network(
        model_list[max_index], data_pro_new, np.array(label_pro_new))
    AC_auc, AC_acc = Test_network(
        model_list[max_index], data_AC, np.array(label_AC))
    AC_auc_new, AC_acc_new = Test_network(
        model_list[max_index], data_AC_new, np.array(label_AC_new))
    PRO34_auc_new, PRO34_acc_new = Test_network(
        model_list[max_index], data_PRO34_new, np.array(label_PRO34_new))

    # log the information
    log_dict = {
        'train_auc': auc_list[max_index],
        'n_feature': len(args['protein_list_input']),
        'ret_auc': ret_auc,
        'ret_acc': ret_acc,
        'pro_auc': pro_auc,
        'pro_acc': pro_acc,
        'pro_auc_new': pro_auc_new,
        'pro_acc_new': pro_acc_new,
        'AC_auc': AC_auc,
        'AC_acc': AC_acc,
        'AC_auc_new': AC_auc_new,
        'AC_acc_new': AC_acc_new,
        'PRO34_auc_new': PRO34_auc_new,
        'PRO34_acc_new': PRO34_acc_new,
    }

    # plot the figures
    print(log_dict)
    if args['plot']:
        plot_dict_dis = Analysis(model_list[max_index], data_dis, np.array(
            label_dis), infostr='Dis', patient_name=patient_dis)
        plot_dict_ret = Analysis(model_list[max_index], data_ret, np.array(
            label_ret), infostr='Ret', patient_name=patient_ret)
        plot_dict_pro = Analysis(model_list[max_index], data_pro_new, np.array(
            label_pro_new), infostr='Pro', patient_name=patient_pro_new)
        plot_dict_pro34 = Analysis(model_list[max_index], data_PRO34_new, np.array(
            label_PRO34_new), infostr='Pro34', patient_name=patient_PRO34_new)

        data_all = np.concatenate([data_ret, data_pro_new])
        label_all = np.concatenate([label_ret, label_pro_new])

        # dict_model_com = ROCModelCompair(
        #     model=model_list[max_index],
        #     train_data=data_dis,
        #     train_label=label_dis,
        #     test_data=data_all,
        #     test_label=label_all)

        log_dict.update(plot_dict_dis)
        log_dict.update(plot_dict_ret)
        log_dict.update(plot_dict_pro)
        log_dict.update(plot_dict_pro34)
        # log_dict.update(dict_model_com)
    if args['use_wandb'] == 1:
        wandb.log(log_dict)
    else:
        path = 'log/{}.csv'.format(args['project'])
        if not os.path.exists(path):
            f = open(path, 'a')
            savelist = [str(item) for item in list(
                args.keys())+list(log_dict.keys())]
            print(','.join(savelist), file=f)
            pass
        else:
            f = open(path, 'a')
            savelist = [str(item).replace(',', '_')
                        for item in list(args.values())+list(log_dict.values())]
            print(','.join(savelist), file=f)

    import pickle
    model = model_list[max_index]
    with open('model_save/some-file.pkl', 'wb') as f:
        pickle.dump(model, f)
    # explainer = shap.Explainer(model)
    # shap_values = explainer.shap_values(data_dis)
    # shap.force_plot(explainer.expected_value, shap_values, data_dis)
    # plt.savefig('zzl.png')


if __name__ == "__main__":

    main()
