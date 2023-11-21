import csv
import os

import pandas as pd


def read_df(csv_name, reattack_list):
    path_ori = os.path.join('../xxxx', csv_name)
    path_reattack = os.path.join('../xxxx', csv_name)
    save_path = os.path.join('../save', csv_name)
    df_ori = pd.read_csv(path_ori)
    df_atk_again = pd.read_csv(path_reattack)
    for i in range(len(reattack_list)):
        df_ori.loc[reattack_list[i]] = df_atk_again.loc[i]
    df_ori.to_csv(save_path, quoting=csv.QUOTE_NONNUMERIC, index=False, mode='a', header=True)

