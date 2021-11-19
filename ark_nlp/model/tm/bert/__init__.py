from ark_nlp.dataset import PairMergeSentenceClassificationDataset as Dataset
from ark_nlp.dataset import PairMergeSentenceClassificationDataset as BertTMDataset

from ark_nlp.processor.tokenizer.transfomer import PairTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import PairTokenizer as BertTMTokenizer

from ark_nlp.nn import BertConfig
from ark_nlp.nn import BertConfig as ModuleConfig

from ark_nlp.nn import Bert
from ark_nlp.nn import Bert as Module

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_bert_optimizer

from ark_nlp.factory.task import TMTask as Task
from ark_nlp.factory.task import TMTask as BertTMTask

from ark_nlp.factory.predictor import TMPredictor as Predictor
from ark_nlp.factory.predictor import TMPredictor as BertTMPredictor
