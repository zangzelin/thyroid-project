import numpy as np
import random as rd
import torch


def SetSeed(seed):
    """function used to set a random seed

    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)


if __name__ == "__main__":
    
    a_list = [
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'P42224', 'P27797', 'Q9HAT2', 'P46940', 'O14964', 'Q8IXM2', 'P17931'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'P42224', 'P27797', 'Q9HAT2', 'P46940', 'O14964', 'Q8IXM2', 'Q01130'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P26038', 'P00568', 'Q13263', 'P15090', 'Q8WX93', 'P42224', 'Q9Y696', 'Q9HAT2', 'P30086', 'O14964', 'P10909', 'P17931'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P23297', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'Q7Z4V5', 'P27797', 'O00170', 'Q12797', 'O14964', 'P09467', 'P17931'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P23297', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'Q7Z4V5', 'P27797', 'Q9BUT1', 'Q12797', 'O14964', 'P09467', 'P17931'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P23297', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'Q7Z4V5', 'P27797', 'Q9Y5X3', 'Q12797', 'O14964', 'P09467', 'P17931'],
        ['P02765', 'P06703', 'O00339', 'P04899', 'P37802', 'P04216', 'P09496', 'P26038', 'P00568', 'P15090', 'Q07654', 'Q6IQ23', 'P25788', 'P50402', 'Q9HAT2', 'P30086', 'O43143', 'P10909', 'P17931'],
        ['P02765', 'P06703', 'O00339', 'P04899', 'P37802', 'P04216', 'P09496', 'P26038', 'P00568', 'P15090', 'Q07654', 'Q6IQ23', 'P25788', 'P50402', 'Q9HAT2', 'P30086', 'O43143', 'P10909', 'P17931'],
        ['P02765', 'P07202', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P26038', 'P00568', 'P78527', 'P04792', 'P35579', 'P42224', 'P27797', 'Q9HAT2', 'P30086', 'P20340', 'P10909', 'P17931'],
        ['P50454', 'P04083', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P06703', 'P00568', 'P78527', 'P04792', 'P35579', 'P42224', 'P27797', 'Q9HAT2', 'P30086', 'O14964', 'P10909', 'P17931'],
        ['P02765', 'P04083', 'O00339', 'P04899', 'O75347', 'P04216', 'P02751', 'P82979', 'P00568', 'P78527', 'P04792', 'P35579', 'P42224', 'P27797', 'Q9HAT2', 'P30086', 'O14964', 'P10909', 'P17931'],
    ]
    b = ["P02765","P04083","O00339","P58546","O75347","P04216","P02751","P83731","P00568","P78527","P04792","P57737","P42224","P27797","Q9HAT2","P30086","O14964","P10909","P17931"]


    for a in a_list:
        c = 0
        for i in a:
            if i in b:
                c += 1
        print(c/len(a))


    # import wandb
    # api = wandb.Api()

    # # run is specified by <entity>/<project>/<run id>
    # run = api.run("zangzelin/TPD2021_v5/1me8zgyk")

    # # save the metrics for the run to a csv file
    # metrics_dataframe = run.history()
    # a1= metrics_dataframe['train_auc']
    # a2= metrics_dataframe['AC_auc_new']
    # a3= metrics_dataframe['ret_acc']
    # a4= metrics_dataframe['pro_acc_new']
    # import matplotlib.pyplot as plt

    # plt.plot(a1, label='train_AUC')
    # plt.plot(a3, label='ret_ACC')
    # plt.plot(a4, label='pro_ACC')
    # plt.plot(a2, label='AC_ACC')
    # plt.legend()
    # plt.savefig('epoch_wish_per.pdf')
    