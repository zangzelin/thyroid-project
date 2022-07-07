import pandas as pd
import numpy as np
import argparse
from sklearn import metrics


def get_metrics(filepath, reference, datatype):
    import pdb
    # pdb.set_trace()
    df = pd.read_csv(filepath)
    label = df['label'].values
    predict_score = df['predict'].values
    predict_binary = (predict_score >= 0.5).astype(np.int_)
    all_accuracy = np.sum(label == predict_binary)/len(label)
    all_auc = metrics.roc_auc_score(label, predict_score)
    if datatype == "pro" or datatype == "prospective":
        df_refer = pd.read_csv(reference)
        data_list = df['patient'].values.tolist()
        label_list = []
        predict_list = []
        predict_binary_list = []
        for i in range(len(data_list)):
            index = np.where(df_refer['Patient_ID'] == data_list[i])[0][0]
            if df_refer['Bethesda_Score'][index] == 'III' or df_refer['Bethesda_Score'][index] == 'IV':
                label_list.append(label[i])
                predict_list.append(predict_score[i])
                predict_binary_list.append(predict_binary[i])
        BS_accuracy = np.sum(np.array(label_list) == np.array(
            predict_binary_list)) / len(label_list)
        BS_auc = metrics.roc_auc_score(
            np.array(label_list), np.array(predict_list))
        print("Result of prospective data:")
        print("All accuracy: %f" % all_accuracy)
        print("All auc: %f" % all_auc)
        print("BS accuracy: %f" % BS_accuracy)
        print("BS auc: %f" % BS_auc)
    elif datatype == "retro" or datatype == "retrospective":
        df_refer = pd.read_csv(reference)
        data_list = df['patient'].values.tolist()
        label_list = []
        predict_list = []
        predict_binary_list = []
        for i in range(len(data_list)):
            index = np.where(df_refer['Patient_ID'] == data_list[i])[0][0]
            if df_refer['Histopathology_type'][index] == 'A' or df_refer['Histopathology_type'][index] == 'C':
                label_list.append(label[i])
                predict_list.append(predict_score[i])
                predict_binary_list.append(predict_binary[i])
        AC_accuracy = np.sum(np.array(label_list) == np.array(
            predict_binary_list))/len(label_list)
        AC_auc = metrics.roc_auc_score(
            np.array(label_list), np.array(predict_list))
        print("Result of retrospective data:")
        print("All accuracy: %f" % all_accuracy)
        print("All auc: %f" % all_auc)
        print("AC accuracy: %f" % AC_accuracy)
        print("AC auc: %f" % AC_auc)
    else:
        print("All accuracy: %f" % all_accuracy)
        print("All auc: %f" % all_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        required=True, help="csv file path")
    parser.add_argument("--reference", type=str,
                        required=True, help="reference file")
    parser.add_argument("--datatype", type=str, default="",
                        help="retro or prop or train")
    args = parser.parse_args()
    get_metrics(args.filepath, args.reference, args.datatype)
