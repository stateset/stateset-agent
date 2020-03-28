import torch
import logger

from pytorch_transformers.tokenization import BertTokenizer
from fast_bert.data import BertDatabunch
from fast_bert.learner import BertLearner
from fast_bert.metrics import accuracy

device = torch.device('cuda')
logger = logging.getLogger()

metrics = [{'name': 'accuracy', 'function': accuracy}]

tokenizer = BertTokenizer.from_pretrained
            ('bert-base-uncased',
            do_lower_case=True)

databunch = BertDatabunch([])

learner = BertLearnder.from_pretrained_model(databunch,)

learner.fit(3, lr='1e-2')