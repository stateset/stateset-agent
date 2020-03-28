# stateset-agent

Agent for calling the Stateset Network to provide a service of question answering and natural language inference.

We will be using XLNet but evaluating all of the state-of-the-art NLP capabilities.

XLNet combines the bidirectional capability of BERT with the autoregressive technology of Transformer-XL:

Like BERT, XLNet uses bidirectional context, which means it looks at the words before and after a given token to predict what it should be. To this end, XLNet maximizes the expected log-likelihood of a sequence with respect to all possible permutations of the factorization order.


As an autoregressive language model, XLNet doesn’t rely on data corruption, and thus avoids BERT’s limitations due to masking – i.e., pretrain-finetune discrepancy and the assumption that unmasked tokens are independent of each other.

To further improve architectural designs for pretraining, XLNet integrates the segment recurrence mechanism and relative encoding scheme of Transformer-XL.


To start, we will be using the Stanford Question Answering Dataset (SQuAD 2.0) for training and evaluating our model. SQuAD is a reading comprehension dataset and a standard benchmark for QA models. The dataset is publicly available on the website.
