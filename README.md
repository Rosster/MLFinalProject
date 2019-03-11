# Using Tweets and DOJ Indictments to Predict Stock Movements

## Project Idea

We plan to apply NLP techniques to extract features from the input text documents, and then build a predictive model that will use those features to forecast changes in stock price.  We will use sentiment analysis and entity recognition to derive the inputs to the model, and then build a regressor to perform the prediction task.  We expect tweets to be a good predictor of stock movement because tweets are often written soon after some sort of newsworthy event that might influence stock price.  As such, they should be strong, proximate indicators of underlying market trends.  Likewise, DOJ indictments are expected to be highly influential, but we will likely have to featurize them in a different way.

## Data Sources

[Tweets](https://www.kaggle.com/davidwallach/financial-tweets)

[Press Releases](https://www.kaggle.com/jbencina/department-of-justice-20092018-press-releases)

[Stocks](https://www.kaggle.com/timoboz/stock-data-dow-jones)

The first two datasets are rich sources of text data that we hypothesize to have influence on stock prices.  The twitter dataset is a curated and tagged set of 25,000 tweets by verified accounts of people and organizations that report on the finance industry.  The tweets are filtered to instances where the user tweeted about one of 584 companies.  The DOJ dataset includes indictments related to publicly traded companies, we will have to filter out other, unrelated filings.  The target dataset includes daily stock price information for all firms’ stocks that are traded on the dow jones industrial index.


TODO:
1. Determine Industries to focus on (DONE)
2. Map Industry to list of stock tickers (DONE)
3. Map DOJ Filing to Industry (DONE)
4. Check feasibility of mapping DOJ filing to company (DONE)
   1. Map Ticker to Full company name and then do simply string includes
5. Use Google Sheets / Google Finance to get the Daily prices
   1. =GOOGLEFINANCE("AAPL", "price", "1/1/2018", "12/31/2018", "DAILY")
   OR USE Google Python API for GoogleFinance (DONE)
6. Parse DOJ Dataset into Dataframe (DONE)
7. Parse above google sheets results into Dataframe (DONE)
8. Is there a stock movement based simply on a mention in the DOJ Filing (DONE)
9. Narrow down to companies that have DOJ related events: if the filing doesn't impact them, it probably doesn't impact anyone else.
10. Determine how to add additional regressors to ARIMA model ([ARIMA-X] or [add ARIMA error as regressor alongside non time-series regressors and check it's coefficient i.e. impact on target])

Models to explore:
Logistic Regression/Classification

NLP on DOJ entry content using spacy


---

Following creation of `doj_data_with_tags_and_industries.json`

Use the following:
```
date
title
clean_orgs
tagged_symbols
tagged_companies
sectors
industries
```


---
Predictive Goal
Target:
- Normalized Stock Price Movement at T

Predictors:

- Normalized Stock Price Movement at T-1
- Normalized Stock Price Movement at T-2
- Normalized Stock Price Movement at T-3

```
"Given yesterday's closing price and todays tweets and filings, what is todays closing price?"
```



Find unique symbols in tagged_symbols  
group records by symbol 
  ^^ a little complicated, basically group if symbol in record.tagged_symbols



If the output will be predicted closing price 
  --> This must be a regression problem since its continous output
- Linear Regression
- Polynomial Fit
  


If we will simply predict whether the closing price will be above or below the opening price 
  --> then we can run classification models

- Decision Tree
- Random Forrest
- Gradient Boosting
- Logistic Regression (Classifier)
- Neural Network
  - Squential
  - Recurrent
- SVC Classifier




For Regression Models 
Input will be moving average of (T-n) days (ordered by date)

Time-series model: ARIMA(3,1,3):

 - **AR**(3) -
 3 days of autocorrelated movements  
- **I**(1) - differenced 1 time 
- **MA**(3) - taking into account the stock's 3 day moving average

We test whether adding the sentiment predictors significantly improves the performance of this naive ARIMA model by measuring reduction in MSE, if any.


For Classification Models
Input will be
- the n columns representing T and [(T-i) for i in range(0, n)] columns (ordered by date)
- sentiment on title and content?

