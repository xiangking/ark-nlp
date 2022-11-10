from ark_nlp.model.seq2seq.unilm.dataset import UniLMDataset as Dataset
from ark_nlp.model.seq2seq.unilm.dataset import UniLMDataset

from ark_nlp.processor.tokenizer.transformer import TransformerTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transformer import TransformerTokenizer as UniLMBertTokenizer

from ark_nlp.nn import BertConfig as ModuleConfig
from ark_nlp.nn import BertConfig as UniLMBertConfig

from ark_nlp.model.seq2seq.unilm.module import UniLMBert as Module
from ark_nlp.model.seq2seq.unilm.module import UniLMBert

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_unilm_bert_optimizer

from ark_nlp.model.seq2seq.unilm.task import UniLMTask as Task
from ark_nlp.model.seq2seq.unilm.task import UniLMTask

# from ark_nlp.model.seq2seq.unilm.predictor import UniLMBertPredictor as Predictor
# from ark_nlp.model.seq2seq.unilm.predictor import UniLMBertPredictor