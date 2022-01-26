from ark_nlp.model.re.afea_bert.afea_relation_extraction_dataset import AFEAREDataset
from ark_nlp.model.re.afea_bert.afea_relation_extraction_dataset import AFEAREDataset as Dataset

from ark_nlp.processor.tokenizer.transfomer import SpanTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import SpanTokenizer as AFEARETokenizer

from ark_nlp.nn import BertConfig as AFEABertConfig
from ark_nlp.nn import BertConfig as ModuleConfig

from ark_nlp.model.re.afea_bert.afea_bert import AFEABert
from ark_nlp.model.re.afea_bert.afea_bert import AFEABert as Module

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_casrel_bert_optimizer

from ark_nlp.model.re.afea_bert.afea_relation_extraction_task import AFEARETask as Task
from ark_nlp.model.re.afea_bert.afea_relation_extraction_task import AFEARETask as AFEAREBert

from ark_nlp.model.re.afea_bert.afea_relation_extraction_predictor import AFEAREPredictor as Predictor
from ark_nlp.model.re.afea_bert.afea_relation_extraction_predictor import AFEAREPredictor as AFEAREPredictor