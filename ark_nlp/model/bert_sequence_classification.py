from ark_nlp.nn import Ernie
from ark_nlp.nn.configuration import ErnieConfig

from ark_nlp.processor.tokenizer import SentenceTokenizer
from ark_nlp.factory.task import SequenceClassificationTask
from ark_nlp.factory.predictor import TCPredictor


class BertSequenceClassificationModel:
    def __init__(
        self,
        model_name,
        dataset,
        **kwargs,
    ):

        MODEL_CLASSES = {
            "nghuyong/ernie-1.0": (ErnieConfig, Ernie, SentenceTokenizer),

        }
        
        self.cat2id = dataset.cat2id
        self.tokenizer = SentenceTokenizer(vocab=MODEL_CLASSES[model_name], max_seq_len=512)
        self.bert_config = MODEL_CLASSES[0].from_pretrained(model_name, num_labels=len(self.cat2id))

        self.module = Ernie.from_pretrained(model_name, config=bert_config)

        self.optimizer = get_default_optimizer(self.module)

        self.model = SequenceClassificationTask(self.module, self.optimizer, 'ce', cuda_device=cuda_device)

        self.predictor = TCPredictor(self.module, self.tokenizer, self.cat2id)

    def fit(
        self, 
        **kwargs
    ):
        self.model.fit(**kwargs)

    def evaluate(
        self, 
        **kwargs
    ):
        self.model.evaluate(**kwargs)

    def predict_one_sample(
        self, 
        **kwargs
    ):
        self.predictor.predict_one_sample(**kwargs)

    def predict_one_sample(
        self, 
        **kwargs
    ):
        self.predictor.predict_batch(**kwargs)