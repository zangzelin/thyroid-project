from sklearn.metrics import roc_curve, auc, recall_score
color = ['#537e35', '#e17832', '#5992c6', '#C00000']
leg = ['B->B', 'M->M', 'B->M', 'M->B']
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import pandas as pd
from sklearn import svm
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import umap


def ROCModelCompair(model, train_data, train_label, test_data, test_label):
    
    def Getpredict(method_name, train_data, train_label, test_data):    
    
        if method_name == 'svm':
            net = svm.SVC(probability=True)
        if method_name == 'RandomForest':
            net = RandomForestClassifier(random_state=100)
        if method_name == 'Lasso':
            net = linear_model.Lasso(alpha=0.02, random_state=100)
        if method_name == 'MLP':
            net = MLPClassifier(random_state=100)
        if method_name == 'LogisticRegression':
            net = LogisticRegression(penalty='none', random_state=100)
        if method_name == 'DecisionTree':
            net = tree.DecisionTreeClassifier(random_state=100)
        
        net.fit(X=train_data, y=train_label)

        if method_name == 'Lasso':
            preValue = net.predict(test_data)
        else:
            preValue = net.predict_proba(test_data)[:, 1]
        return preValue
    
    Xval = torch.tensor(test_data).float()
    y_pre_our = model.predict_proba(Xval)[:,1]

    fig = plt.figure(figsize=(6, 6))
    for m in ['svm', 'RandomForest', 'Lasso', 'MLP', 'LogisticRegression', 'DecisionTree', 'our']:
        if m == 'our':
            pre = y_pre_our
        else:
            pre = Getpredict(m, train_data, train_label, test_data)
        fpr, tpr, threshold = roc_curve(test_label, pre, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # paid = paid.to_list()
        plt.plot(fpr, tpr, label='%s ROC curve ( area = %.4f )' % (
            m, roc_auc))
    
    plt.plot([0, 1], [0, 1], color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ModelCompair.pdf', dpi=500)
    return {'ModelCompair':wandb.Image(fig)}


def ACC(predict, label,  best_threshold=0.5, infostr='',):

    leg = ['B->B', 'M->M', 'B->M', 'M->B']

    def GetACCwithThre(threshold_candi):

        predict_bool = predict.copy()
        predict_bool[predict_bool > threshold_candi] = 1
        predict_bool[predict_bool < threshold_candi] = 0

        is_wrong = (predict_bool != label)
        ACC_candi = 1-is_wrong.sum()/len(predict)
        return ACC_candi, is_wrong


    ACC, is_wrong = GetACCwithThre(best_threshold)
    color_type = label + is_wrong*2

    fig = plt.figure(figsize=(6, 6))
    y_all = np.array([i for i in range(len(predict))])
    for i in range(4):
        index = color_type == (i)

        x = predict[index]
        y = y_all[index]
        plt.scatter(x, y, label=leg[i], color=color[i])
    plt.title('pic/ACC_{}_{}'.format(infostr,ACC))
    plt.savefig('log/ACC_{}.pdf'.format(infostr))
    
    return fig


def Roc(predict, label, infostr=''):

    
    fpr, tpr, threshold = roc_curve(label, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # paid = paid.to_list()
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color=color[0], label='%s ROC curve ( area = %.4f )' % (
        infostr, roc_auc))
    plt.plot([0, 1], [0, 1], color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('pic/ROC_{}'.format(infostr))
    plt.legend()
    plt.savefig('log/ROC_{}.pdf'.format(infostr))

    return fig

def ScatterTsne(emb, label, infostr=''):


    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x=emb[:,0], y=emb[:,1], c=label)
    plt.title('pic/TSNE_{}'.format(infostr))
    plt.legend()

    plt.savefig('log/TSNE_{}.pdf'.format(infostr))

    return fig

def ScatterUMAP(emb, label, infostr=''):


    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x=emb[:,0], y=emb[:,1], c=label)
    plt.title('pic/UMAP_{}'.format(infostr))
    plt.legend()

    plt.savefig('log/UMAP_{}.pdf'.format(infostr))

    return fig

def Analysis(net, data, label, infostr, patient_name=None):
    Xval = torch.tensor(data).float()
    yval = label
    y_pre = net.predict_proba(Xval)[:,1]
    
    latent = net.predict_latent(Xval)
    emb_tsne = TSNE(random_state=0).fit_transform(latent)
    emb_umap = umap.UMAP(random_state=0,  n_neighbors=30).fit_transform(latent)
        # print(meta_data_pd)

    plot_dict = {
        '{}_ROC'.format(infostr) : wandb.Image(Roc(y_pre, yval, infostr=infostr)),
        '{}_ACC'.format(infostr) : wandb.Image(ACC(y_pre, yval, infostr=infostr, best_threshold=0.5)),
        '{}_TSNE'.format(infostr) : wandb.Image(ScatterTsne(emb_tsne, yval, infostr=infostr)),
        '{}_UMAP'.format(infostr) : wandb.Image(ScatterUMAP(emb_umap, yval, infostr=infostr)),
    }
    if patient_name is not None:
        meta_data_pd = pd.DataFrame(
            np.concatenate(
                [y_pre[:,None], yval[:,None], emb_tsne[:,0][:,None], emb_tsne[:,1][:,None], emb_umap[:,0][:,None], emb_umap[:,1][:,None]], 
                axis=1
                ),
            index=patient_name, 
            columns=['predict', 'label', 'tsne1', 'tsne2', 'umap1', 'umap2']
            )
        meta_data_pd['patient'] = patient_name
        
        meta_data_pd.to_csv('log/{}_metadata.csv'.format(infostr))
        wandb.save('log/{}_metadata.csv'.format(infostr))
            
        plot_dict['{}_metadata'.format(infostr)] = wandb.Table(dataframe=meta_data_pd)
    
    return plot_dict
