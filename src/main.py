from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from model_generators import load_rnn_timeser_preds


class Runner(object):
    def __init__(self):
        self.start = datetime.now()
        self.train_df = None

    def load_data(self):
        self.doj_df = pd.read_json(
            "./data/doj_data_with_tags_and_industries_and_sentiment.json"
        )
        self.doj_df = self.doj_df.reset_index()

        self.twitter_df = pd.read_json("./data/twitter_data_with_sentiment.json")
        self.twitter_df = self.twitter_df.reset_index()

        self.stock_df = pd.read_csv("./data/stock_doj_mention_only.csv")
        self.stock_df = self.stock_df.reset_index()
        return self

    def generate_prepared_df(self, data_to_look_at):
        self.data_to_look_at = data_to_look_at
        if not data_to_look_at:
            raise ValueError("missing data_to_look_at")

        self.prepare_df()
        with open("./data/prepared_df_{}.json".format(data_to_look_at), "w") as f:
            f.write(self.train_df.to_json())
        return self

    def load_prepared_df(self, name):
        self.train_df = pd.read_json("./data/{}".format(name))
        self.train_df = self.train_df.reset_index()
        return self

    def prepare_df(self):
        # call this independently to generate the prepared_df json file
        """
      DF columns
      date (in ms), symbol, doj_entries, doj_sentiment, sectors, industries,  t-1, t-2, t-3, closing price

      response variable is closing price
      response variable for classification, +25%, +50, +75, -25, -50, -75
    """
        symbols = list(self.stock_df.columns)[
            1:
        ]  # dont include first column, which is date
        output_columns = {
            "date": [],
            "symbol": [],
            "t-1": [],
            "t-2": [],
            "t-3": [],
            "closing_price": [],
        }

        if self.data_to_look_at == "doj":
            output_columns["doj_entries"] = []
            output_columns["doj_sentiment"] = []
        elif self.data_to_look_at == "twitter":
            output_columns["twitter_entries"] = []
            output_columns["twitter_sentiment"] = []

        total = len(self.stock_df)
        for index in range(0, total):
            print("%d/%d" % (index, total))
            row = self.stock_df.iloc[index]
            date_obj, date_ms = date_to_time_in_ms(row["date"])
            for symbol in symbols:
                close_price_for_symbol = row[symbol]
                output_columns["date"].append(date_ms)
                output_columns["symbol"].append(symbol)

                if self.data_to_look_at == "doj":
                    relevant_records_df = self.get_relevant_doj_records(
                        date_obj=date_obj, symbol=symbol
                    )
                    output_columns["doj_entries"].append(len(relevant_records_df))
                    output_columns["doj_sentiment"].append(
                        self.get_sentiment(relevant_records_df=relevant_records_df)
                    )
                elif self.data_to_look_at == "twitter":
                    relevant_records_df = self.get_relevant_twitter_records(
                        date_obj=date_obj, symbol=symbol
                    )
                    output_columns["twitter_entries"].append(len(relevant_records_df))
                    output_columns["twitter_sentiment"].append(
                        self.get_sentiment(relevant_records_df=relevant_records_df)
                    )

                for n in range(1, 4):
                    append_historical_closing(
                        n=n,
                        output_columns=output_columns,
                        index=index,
                        close_price_for_symbol=close_price_for_symbol,
                    )
                output_columns["closing_price"].append(close_price_for_symbol)

        self.train_df = pd.DataFrame.from_dict(output_columns)
        return self

    def get_relevant_doj_records(self, date_obj, symbol):
        lower_bound = date_obj.strftime("%Y-%m-%d")
        upper_bound = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        date_mask = (self.doj_df["date"] >= lower_bound) & (
            self.doj_df["date"] < upper_bound
        )
        within_timeframe_df = self.doj_df.loc[date_mask]
        symbol_mask = within_timeframe_df["tagged_symbols"].apply(lambda x: symbol in x)
        return within_timeframe_df[symbol_mask]

    def get_relevant_twitter_records(self, date_obj, symbol):
        lower_bound = date_obj.strftime("%Y-%m-%d")
        upper_bound = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        date_mask = (self.twitter_df["timestamp"] >= lower_bound) & (
            self.twitter_df["timestamp"] < upper_bound
        )
        within_timeframe_df = self.twitter_df.loc[date_mask]
        symbol_mask = within_timeframe_df["symbols"].apply(
            lambda x: symbol in x.lower()
        )
        return within_timeframe_df[symbol_mask]

    def get_sentiment(self, relevant_records_df):
        """
      sentiment_neg                                                       0.043
      sentiment_neu                                                       0.865
      sentiment_pos                                                       0.092
    """
        sentiments = []
        for row in relevant_records_df.itertuples():
            stats = {
                0: getattr(row, "sentiment_neg"),
                0.5: getattr(row, "sentiment_neu"),
                1: getattr(row, "sentiment_pos"),
            }
            sentiments.append(max(stats.keys(), key=(lambda key: stats[key])))
        return np.average(sentiments)

    def prepare_df_for_classification(self, prepare_df_from_regression):
        return self

    def end(self):
        now = datetime.now()
        print("Time Elapsed: ", now - self.start)


def date_to_time_in_ms(date_str):
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    return date_obj, int(date_obj.strftime("%s")) * 1000


def append_historical_closing(n, output_columns, index, close_price_for_symbol):
    if index - n >= 0:
        output_columns["t-%s" % n].append(output_columns["closing_price"][index - n])
    else:
        output_columns["t-%s" % n].append(close_price_for_symbol)  # ramp up


if __name__ == "__main__":
    # Env: /Users/rohanjyoti/virtual_envs/mlenv/bin/python3
    #   point vscode env to this, settings.json -> workspace settings
    #   "python.pythonPath": "/Users/rohanjyoti/virtual_envs/mlenv/bin/python3"
    # To Run: time ~/virtual_envs/mlenv/bin/python3 main.py
    # x = Runner().load_data().load_prepared_df("prepared_df_twitter.json").end()
    # x = Runner().load_data().load_prepared_df("prepared_df_twitter.json")
    runner = Runner().load_data().load_prepared_df("prepared_df_doj.json")
    df = runner.train_df
    runner2 = Runner().load_data().load_prepared_df("prepared_df_twitter.json")
    df2 = runner2.train_df

rnn_preds_df, rnn_obs_df = load_rnn_timeser_preds.get_rnn_preds()
print(rnn_preds_df.head(), rnn_obs_df.head())
"""
from main import Runner
runner = Runner().load_data().generate_prepared_df(data_to_look_at='doj')
runner2 = Runner().load_data().generate_prepared_df(data_to_look_at='twitter')

runner = Runner().load_data().load_prepared_df('prepared_df_doj.json')
df = runner.train_df
runner2 = Runner().load_data().load_prepared_df('prepared_df_twitter.json')
df2 = runner2.train_df

"""

