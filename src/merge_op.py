df_all = df.merge(
    rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True
).symbol.unique()

rnn_preds_df[
    rnn_preds_df["symbol"].isin(
        df.merge(
            rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True
        ).symbol.unique()
    )
]


df[
    df.merge(rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True)[
        "_merge"
    ]
    == "both"
].symbol.unique()

len(
    df[
        df.merge(
            rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True
        )["_merge"]
        == "both"
    ].symbol.unique()
)

len(
    df[
        df.merge(
            rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True
        )["_merge"]
        == "left_only"
    ].symbol.unique()
)

len(df.symbol.unique())


pd.concat(
    [
        df2[df2["symbol"] == "aaba"].sort_values(by=["date"]),
        rnn_preds_df[rnn_preds_df["symbol"] == "aaba"].pred_pct_chg,
    ],
    axis=1,
    ignore_index=True,
)


df_all = df.merge(
    rnn_preds_df.drop_duplicates(), on=["symbol"], how="left", indicator=True
)

# get symbols that appear across all three datasets
intersection_list = []
for x in rnn_preds_df["symbol"]:
    if x in df["symbol"].unique() and x in df2["symbol"].unique():
        intersection_list.append(x)


# Twitter_data
for unique_symbol in rnn_preds_df["symbol"].unique():

    x = df2[df2["symbol"] == unique_symbol].sort_values(
        by=["date"]
    )  # sort by date first
    x.index = range(x.shape[0])  # give new index

    y = rnn_preds_df[rnn_preds_df["symbol"] == unique_symbol]  # assume already sorted
    y.index = range(
        y.shape[0]
    )  # this index should be the same as x if dimensions and sort are same (as assumed)

    # result = x.join(y)  # ta-da
    result = x.join(y, rsuffix="_extra")
    del result["symbol_extra"]

# DOJ data
for unique_symbol in rnn_preds_df["symbol"].unique():

    x = df[df["symbol"] == unique_symbol].sort_values(by=["date"])  # sort by date first
    x.index = range(x.shape[0])  # give new index

    y = rnn_preds_df[rnn_preds_df["symbol"] == unique_symbol]  # assume already sorted
    y.index = range(
        y.shape[0]
    )  # this index should be the same as x if dimensions and sort are same (as assumed)

    # result = x.join(y)  # ta-da
    result = x.join(y, rsuffix="_extra")
    print(result)
    del result["symbol_extra"]

intersection_list = []
for x in rnn_preds_df["symbol"]:
    if x in df["symbol"].unique() and x in df2["symbol"].unique():
        intersection_list.append(x)
intersection_list
