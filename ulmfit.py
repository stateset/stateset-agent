import fast.ai from fast.ai



data_lm = (TextList
          .from_csv("./stateset-network-data/",
          'train-processed.csv', cols=5,
          vocab=data_lm.vocab)
          .split_by_rand_pct()
          .label_from_df(cols=0)
          .databunch())

learn = language_model_learner(data_lm AWD_LSTM, drop_mult=0.3)

learn.lr_find()
learn.recorder.plot()