from ark_nlp.dataset import PairSentenceClassificationDataset as Dataset
from ark_nlp.dataset import PairSentenceClassificationDataset as BertTCDataset

from ark_nlp.processor.tokenizer.transfomer import PairTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import PairTokenizer as BertTMTokenizer

from ark_nlp.nn import BertConfig as BertConfig
from ark_nlp.nn import Bert

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_bert_optimizer

from ark_nlp.factory.task import SequenceClassificationTask as Task
from ark_nlp.factory.task import SequenceClassificationTask as BertTCTask

from ark_nlp.factory.predictor import TMPredictor as Predictor
from ark_nlp.factory.predictor import TMPredictor as BertTCPredictor