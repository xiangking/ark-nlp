from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_dataset import PromptUIEDataset as Dataset
from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_dataset import PromptUIEDataset

from ark_nlp.processor.tokenizer.transfomer import TransfomerTokenizer as Tokenizer
from ark_nlp.processor.tokenizer.transfomer import TransfomerTokenizer as PromptUIETokenizer

from ark_nlp.nn import BertConfig as PromptUIEConfig
from ark_nlp.nn import BertConfig as ModuleConfig

from ark_nlp.model.ie.prompt_uie.prompt_uie import PromptUIE
from ark_nlp.model.ie.prompt_uie.prompt_uie import PromptUIE as Module

from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_model_optimizer
from ark_nlp.factory.optimizer import get_default_bert_optimizer as get_default_prompt_uie_optimizer

from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_task import PromptUIETask as Task
from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_task import PromptUIETask

from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_predictor import PromptUIEPredictor as Predictor
from ark_nlp.model.ie.prompt_uie.prompt_uie_information_extraction_predictor import PromptUIEPredictor
