

from numpy.core.fromnumeric import choose
from sklearn import datasets
# from main import main
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn


def Splitdata(patient_all, label_all, protein_all, set_all, splitstr):

    choose_bool = [True if set_item ==
                   splitstr else False for set_item in set_all]
    # print(choose_bool)

    patient_new = []
    label_new = []

    for i in range(len(set_all)):
        if choose_bool[i] == True:
            patient_new.append(patient_all[i])
            label_new.append(label_all[i])

    protein_new = protein_all[choose_bool, :]
    return patient_new, label_new, protein_new


def Loaddata(path, fillnan=10):
    data_frame = pd.read_csv(path)
    patient_all = data_frame['Unnamed: 0'].tolist()
    label_all_str = data_frame['label'].tolist()
    label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
    set_all = data_frame['Sets'].tolist()
    protein_all = data_frame.drop(
        ['Unnamed: 0', 'label', 'Sets'], axis=1).to_numpy()
    protein_name = data_frame.columns.to_list()
    protein_name.remove('label')
    protein_name.remove('Unnamed: 0')
    protein_name.remove('Sets')

    patient_dis, label_dis, protein_dis = Splitdata(
        patient_all, label_all, protein_all, set_all, 'Discovery')
    patient_ret, label_ret, protein_ret = Splitdata(
        patient_all, label_all, protein_all, set_all, 'Retrospective Test')
    patient_pro, label_pro, protein_pro = Splitdata(
        patient_all, label_all, protein_all, set_all, 'Prospective Test')

    return patient_dis, label_dis, protein_dis, patient_ret, label_ret, protein_ret, patient_pro, label_pro, protein_pro, protein_name


def LoaddataProteincutoff(path, cutoff=0.6):
    data_frame = pd.read_csv(path)
    patient_all = data_frame['Unnamed: 0'].tolist()
    label_all_str = data_frame['label'].tolist()
    label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
    set_all = data_frame['Sets'].tolist()
    dis_bool = data_frame['Sets'] == 'Discovery'
    data_dis = data_frame.loc[dis_bool].drop(
        ['Unnamed: 0', 'label', 'Sets'], axis=1)
    na_rate = data_dis.isna().sum()/data_dis.shape[0]
    nacut = na_rate[na_rate < cutoff]

    return nacut.index.tolist()


def LoaddataAC(path, fillnan=10):
    data_frame = pd.read_csv(path)
    patient_all = data_frame['Unnamed: 0'].tolist()
    label_all_str = data_frame['label'].tolist()
    label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
    set_all = data_frame['Sets'].tolist()
    protein_all = data_frame.drop(
        ['Unnamed: 0', 'label', 'Sets', 'Histopathology_type'], axis=1).to_numpy()
    protein_name = data_frame.columns.to_list()
    protein_name.remove('label')
    protein_name.remove('Unnamed: 0')
    protein_name.remove('Sets')
    protein_name.remove('Histopathology_type')

    Histopathology_type_list = data_frame['Histopathology_type'].tolist()
    choose_bool = [True if (set_item in 'AC' and set_all[i] != 'Discovery')
                   else False for i, set_item in enumerate(Histopathology_type_list)]
    patient_new = []
    label_new = []
    for i in range(len(set_all)):
        if choose_bool[i] == True:
            patient_new.append(patient_all[i])
            label_new.append(label_all[i])
    protein_new = protein_all[choose_bool, :]

    # patient_dis, label_dis, protein_dis = Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
    # patient_ret, label_ret, protein_ret = Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
    # patient_pro, label_pro, protein_pro = Splitdata(patient_all, label_all, protein_all, set_all, 'Prospective Test')

    return patient_new, label_new, protein_new


def LoaddataPRO34(path, fillnan=10):
    data_frame = pd.read_csv(path)
    patient_all = data_frame['Unnamed: 0'].tolist()
    label_all_str = data_frame['label'].tolist()
    label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
    set_all = data_frame['Sets'].tolist()
    protein_all = data_frame.drop(
        ['Unnamed: 0', 'label', 'Sets', 'Histopathology_type', 'Bethesda_Score'], axis=1).to_numpy()
    protein_name = data_frame.columns.to_list()
    protein_name.remove('label')
    protein_name.remove('Unnamed: 0')
    protein_name.remove('Sets')
    protein_name.remove('Histopathology_type')

    Histopathology_type_list = data_frame['Bethesda_Score'].tolist()
    choose_bool = [True if ((set_item == 'III') or (set_item == 'IV') and set_all[i] ==
                            'Prospective Test') else False for i, set_item in enumerate(Histopathology_type_list)]
    patient_new = []
    label_new = []
    for i in range(len(set_all)):
        if choose_bool[i] == True:
            patient_new.append(patient_all[i])
            label_new.append(label_all[i])
    protein_new = protein_all[choose_bool, :]

    # patient_dis, label_dis, protein_dis = Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
    # patient_ret, label_ret, protein_ret = Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
    # patient_pro, label_pro, protein_pro = Splitdata(patient_all, label_all, protein_all, set_all, 'Prospective Test')

    return patient_new, label_new, protein_new


def Get_protein_index(protein_dict, protein_list):
    index_list = []
    for item in protein_list:
        index_list.append(protein_dict.index(item))
    return index_list


def Get_selected_protein_matrix(proteins, selected_protein_index):
    return proteins[:, selected_protein_index]


def Fill_nan(data, fill_value):

    data[np.isnan(data)] = fill_value
    return data


def StanFeature(data):

    # print(data[:,-1:])
    # print(data[:,-1])
    data_new = data.copy()/data[:, -1:]

    return np.concatenate([data, data_new], axis=1)


if __name__ == "__main__":

    patient_dis, label_dis, protein_dis, patient_ret, label_ret, protein_ret, patient_pro, label_pro, protein_pro = Loaddata(
        'data/alldata.csv')

    print(patient_dis)
    print(label_dis)
    print(protein_dis)
    print(patient_ret)
    print(label_ret)
    print(protein_ret)
    print(patient_pro)
    print(label_pro)
    print(protein_pro)
