import os
import pandas as pd
import sys


def get_rnn_preds():
    prev_working_dir = os.getcwd()
    os.chdir(
        "C:\\Users\\Sky1\PycharmProjects\\backupFinal\\repository\\src\\data\\Stock_Predictions\\"
    )
    x = os.listdir()
    pred_result_list = []
    obs_result_list = []
    # use a list for performance (per pandas docs)
    # pred_cols = ["pred_pct_chg"]
    # obs_cols = ["obs_pct_chg"]

    for filename in x:
        parsed_symbol = filename.split("_")[0]
        if "predicted" in filename:
            pred_rnn_df = pd.read_csv(filename, index_col=None, header=None)
            pred_rnn_df["symbol"] = parsed_symbol
            pred_result_list.append(pred_rnn_df.head(145))
        elif "observed" in filename:
            obs_rnn_df = pd.read_csv(filename, index_col=None, header=None)
            obs_rnn_df["symbol"] = parsed_symbol
            obs_result_list.append(obs_rnn_df.head(145))
        else:
            sys.exit("error. check filenames")
    pred_masterdataframe = pd.concat(pred_result_list, axis=0, ignore_index=True)
    observed_masterdataframe = pd.concat(obs_result_list, axis=0, ignore_index=True)
    pred_masterdataframe.columns = ["pred_pct_chg", "symbol"]
    observed_masterdataframe.columns = ["obs_pct_chg", "symbol"]
    os.chdir(prev_working_dir)

    return pred_masterdataframe, observed_masterdataframe
