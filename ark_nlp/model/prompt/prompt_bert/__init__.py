from ark_nlp.dataset import PromptDataset as Dataset
from ark_nlp.dataset import PromptDataset as PromptBertDataset

from ark_nlp.processor.tokenizer.transfomer import PromptMLMTransformerTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import PromptMLMTransformerTokenizer as PromptBertTokenizer
from ark_nlp.processor.tokenizer.transfomer import PromptMLMTransformerTokenizer

from ark_nlp.nn import BertConfig as PromptBertConfig
from ark_nlp.nn import BertConfig as ModuleConfig

from ark_nlp.nn import BertForPromptMaskedLM as PromptBert
from ark_nlp.nn import BertForPromptMaskedLM as Module

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_prompt_bert_optimizer

from ark_nlp.factory.task import PromptMLMTask as Task
from ark_nlp.factory.task import PromptMLMTask as PromptBertMLMTask

from ark_nlp.factory.predictor import PromptMLMPredictor as Predictor
from ark_nlp.factory.predictor import PromptMLMPredictor as PromptBertMLMPredictor
