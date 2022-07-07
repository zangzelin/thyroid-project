# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import pandas as pd

data_lab = pd.read_csv('data_Jul1/TPD_diann_protMatrix_20210701_lab.csv')
data_dis = pd.read_csv('data_Jul1/TPD_diann_protMatrix_20210701_dis.csv')
data_pro = pd.read_csv('data_Jul1/TPD_diann_protMatrix_20210701_pro.csv')
data_ret = pd.read_csv('data_Jul1/TPD_diann_protMatrix_20210701_ret.csv')

data_dis_indexby_MS_file_name = data_dis.set_index('MS_file_name')
data_pro_indexby_MS_file_name = data_pro.set_index('MS_file_name')
data_ret_indexby_MS_file_name = data_ret.set_index('MS_file_name')

# %%
patients_d = data_lab.sort_values(by=['Sets','Patient_ID'])['Patient_ID'].to_list()
label_d = data_lab.sort_values(by=['Sets','Patient_ID'])['Classificaiton_type'].to_list()
Histopathology_type_d = data_lab.sort_values(by=['Sets','Patient_ID'])['Histopathology_type'].to_list()
Bethesda_Scores_d = data_lab.sort_values(by=['Sets','Patient_ID'])['Bethesda_Score'].to_list()
patients = []
labels = []
Bethesda_Scores = []
Histopathology_type_list = []
for i, p in enumerate(patients_d):
    if p not in patients:
        patients.append(p)
        labels.append(label_d[i])
        Bethesda_Scores.append(Bethesda_Scores_d[i])
        Histopathology_type_list.append(Histopathology_type_d[i])

# %%
mean_list  = []
rename_dict = {}

i=0
for p in patients:
    MS_file_name_for_Patient = data_lab.loc[data_lab['Patient_ID']==p]['MS_file_name'].to_list()
    sets_for_Patient = data_lab.loc[data_lab['Patient_ID']==p]['Sets'].to_list()[0]
    print(i, MS_file_name_for_Patient, p)
    if sets_for_Patient == 'Prospective Test':
        mean = pd.DataFrame(data_pro_indexby_MS_file_name.loc[MS_file_name_for_Patient].mean()).T
    elif sets_for_Patient == 'Discovery':
        mean = pd.DataFrame(data_dis_indexby_MS_file_name.loc[MS_file_name_for_Patient].mean()).T
    elif sets_for_Patient == 'Retrospective Test':
        mean = pd.DataFrame(data_ret_indexby_MS_file_name.loc[MS_file_name_for_Patient].mean()).T
    mean['Sets'] = sets_for_Patient
    mean['label'] = labels[i]
    mean['Bethesda_Score'] = Bethesda_Scores[i]
    mean['Histopathology_type'] = Histopathology_type_list[i]
    mean_list.append(mean)
    rename_dict[i] = p
    i += 1
out = pd.concat(mean_list).reset_index().rename(index=rename_dict).drop(['index'], axis=1)

# %%
out.to_csv('data_Jul1/alldata_Histopathology_type_list0704.csv')

