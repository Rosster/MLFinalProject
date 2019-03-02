# Using Tweets and DOJ Indictments to Predict Stock Movements

## Project Idea

We plan to apply NLP techniques to extract features from the input text documents, and then build a predictive model that will use those features to forecast changes in stock price.  We will use sentiment analysis and entity recognition to derive the inputs to the model, and then build a regressor to perform the prediction task.  We expect tweets to be a good predictor of stock movement because tweets are often written soon after some sort of newsworthy event that might influence stock price.  As such, they should be strong, proximate indicators of underlying market trends.  Likewise, DOJ indictments are expected to be highly influential, but we will likely have to featurize them in a different way.

## Data Sources

[Tweets](https://www.kaggle.com/davidwallach/financial-tweets)

[Press Releases](https://www.kaggle.com/jbencina/department-of-justice-20092018-press-releases)

[Stocks](https://www.kaggle.com/timoboz/stock-data-dow-jones)

The first two datasets are rich sources of text data that we hypothesize to have influence on stock prices.  The twitter dataset is a curated and tagged set of 25,000 tweets by verified accounts of people and organizations that report on the finance industry.  The tweets are filtered to instances where the user tweeted about one of 584 companies.  The DOJ dataset includes indictments related to publicly traded companies, we will have to filter out other, unrelated filings.  The target dataset includes daily stock price information for all firms’ stocks that are traded on the dow jones industrial index.