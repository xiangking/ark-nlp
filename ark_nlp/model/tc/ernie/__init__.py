from ark_nlp.dataset import SentenceClassificationDataset as Dataset
from ark_nlp.dataset import SentenceClassificationDataset as ErnieTCDataset

from ark_nlp.processor.tokenizer.transfomer import SentenceTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import SentenceTokenizer as ErnieTCTokenizer

from ark_nlp.nn import ErnieConfig
from ark_nlp.nn import ErnieConfig as ModuleConfig

from ark_nlp.nn import Ernie
from ark_nlp.nn import Ernie as Module

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_ernie_optimizer

from ark_nlp.factory.task import SequenceClassificationTask as Task
from ark_nlp.factory.task import SequenceClassificationTask as ErnieTCTask

from ark_nlp.factory.predictor import TCPredictor as Predictor
from ark_nlp.factory.predictor import TCPredictor as ErnieTCPredictor