from ark_nlp.dataset import BIONERDataset as Dataset
from ark_nlp.dataset import BIONERDataset as CRFErnieNERDataset

from ark_nlp.processor.tokenizer.transfomer import TokenTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import TokenTokenizer as CRFErnieNERTokenizer

from ark_nlp.nn import ErnieConfig as CRFErnieConfig
from ark_nlp.model.ner.crf_ernie.crf_ernie import CRFErnie

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_crf_ernie_optimizer

from ark_nlp.factory.task import BIONERTask as Task
from ark_nlp.factory.task import BIONERTask as CRFErnieNERTask

from ark_nlp.factory.predictor import BIONERPredictor as Predictor
from ark_nlp.factory.predictor import BIONERPredictor as CRFErniePredictor