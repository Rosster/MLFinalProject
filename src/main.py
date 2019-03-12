from datetime import datetime, timedelta 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder


class Runner(object):
  def __init__(self):
    self.start = datetime.now()
    self.train_df = None


  def load_data(self):
    self.doj_df = pd.read_json('./data/doj_data_with_tags_and_industries.json')
    self.doj_raw = json.loads(self.doj_df.to_json())
    self.stock_df = pd.read_csv('./data/stock_doj_mention_only.csv')
    return self


  def prepare_df(self):
    cols = ['date', 'title', 'clean_orgs', 'tagged_symbols', 'tagged_companies', 'sectors', 'industries']
    
    '''
      DF columns
      date (in ms), symbol, doj_entries, doj_sentiment, sectors, industries,  t-1, t-2, t-3, closing price

      response variable is closing price
      response variable for classification, +25%, +50, +75, -25, -50, -75
    '''

    symbols = list(self.stock_df.columns)[1:] # dont include first column, which is date
    output_columns = {
      'date': [],
      'symbol': [],
      'doj_entries': [],
      'doj_sentiment': [],
      't-1': [],
      't-2': [],
      't-3': [],
      'closing_price': []
    }

    for index, row in self.stock_df.itertuples():
      date_obj, date_ms = date_to_time_in_ms(row['date'])
      for symbol in symbols:
        close_price_for_symbol = row[symbol]
        output_columns['date'].append(date_ms)
        output_columns['symbol'].append(symbol)
        
        relevant_records_df = self.get_relevant_doj_records(date_obj=date_obj, symbol=symbol)
        output_columns['doj_entries'].append(len(relevant_records_df))
        output_columns['doj_sentiment'].append(1) # todo
        
        for n in range(1, 4):
          append_historical_closing(
            n=n, output_columns=output_columns, 
            index=index, close_price_for_symbol=close_price_for_symbol
          )
        output_columns['closing_price'].append(close_price_for_symbol)

    return self



  def get_relevant_doj_records(self, date_obj, symbol):
    lower_bound = date_obj.strftime('%Y-%m-%d')
    upper_bound = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    date_mask = (self.doj_df['date'] >= lower_bound ) & (self.doj_df['date'] < upper_bound)
    within_timeframe_df = self.doj_df.loc[date_mask]
    symbol_mask = within_timeframe_df['tagged_symbols'].apply(lambda x: symbol in x)
    return within_timeframe_df[symbol_mask]
    



  def prepare_df_for_classification(self, prepare_df_from_regression):
    return self



  def end(self):
    now = datetime.now()
    print("Time Elapsed: ", now - self.start)




def date_to_time_in_ms(date_str):
  date_obj = datetime.strptime(date_str, '%m/%d/%Y')
  return date_obj, int(date_obj.strftime("%s")) * 1000


def append_historical_closing(n, output_columns, index, close_price_for_symbol):
  if index - n >= 0:
    output_columns['t-%s' % n].append(output_columns['closing_price'][index-1])
  else:
    output_columns['t-%s' % n].append(close_price_for_symbol) # ramp up



if __name__ == "__main__":
  # Env: /Users/rohanjyoti/virtual_envs/mlenv/bin/python3 
  #   point vscode env to this, settings.json -> workspace settings
  #   "python.pythonPath": "/Users/rohanjyoti/virtual_envs/mlenv/bin/python3"
  # To Run: time ~/virtual_envs/mlenv/bin/python3 main.py
  Runner().load_data()\
          .end()

'''
from main import Runner
runner = Runner().load_data()

df = runner.doj


'''
  