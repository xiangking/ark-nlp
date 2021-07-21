from ark_nlp.dataset import SpanNERDataset as Dataset 
from ark_nlp.dataset import SpanNERDataset as SpanErnieNERDataset

from ark_nlp.processor.tokenizer.transfomer import TokenTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import TokenTokenizer as SpanErnieNERTokenizer

from ark_nlp.nn import BertConfig as SpanErnieConfig
from ark_nlp.model.ner.span_ernie.span_ernie import SpanIndependenceErnie as SpanErnie

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_span_ernie_optimizer

from ark_nlp.factory.task import SpanNERTask as Task
from ark_nlp.factory.task import SpanNERTask as SpanErnieNERTask

from ark_nlp.factory.predictor import SpanNERPredictor as Predictor
from ark_nlp.factory.predictor import SpanNERPredictor as SpanErnieNERPredictor