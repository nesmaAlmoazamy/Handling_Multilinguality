from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from typing import Dict
import logging
import csv
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
EMBEDDING_DIM = 128
HIDDEN_DIM = 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@DatasetReader.register("data-reader")
class MultilingualDatasetReader(DatasetReader):
    def __init__(self,    
        lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy = lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(file_path, "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter=',')
            for row in tsv_in:
                if len(row) == 2:
                    Instance = self.text_to_instance( article=row[1],label=row[0])
                    yield Instance

    @overrides
    def text_to_instance(self,  # type: ignore
                 article: str,
                 label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_article = self._tokenizer.tokenize(article)
        fields["tokens"] = TextField(tokenized_article, self._token_indexers)
#        fields["tokens"] = article
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
