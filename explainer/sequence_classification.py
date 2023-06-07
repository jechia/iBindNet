import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding

from explainer.explainer import BaseExplainer

from explainer.attributions import LIGAttributions


SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class SequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.

    Calculates attribution for `text` using the given model
    and tokenizer.

    Attributions can be forced along the axis of a particular output index or class name.
    To do this provide either a valid `index` for the class label's output or if the outputs
    have provided labels you can pass a `class_name`.

    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.


    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        attribution_type: str = "lig",
        custom_labels: Optional[List[str]] = None,
        pretrain=False,
    ):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.

        Raises:
            AttributionTypeNotSupportedError:
        """
        super().__init__(model, tokenizer,device,pretrain)

        self.attribution_type = attribution_type

        if custom_labels is not None:
            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(custom_labels)

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.n_steps = 50

    @staticmethod
    def _get_id2label_and_label2id_dict(
        labels: List[int],
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2label: Dict[int, str] = dict()
        label2id: Dict[str, int] = dict()
        for idx, label in enumerate(labels):
            id2label[idx] = label
            label2id[label] = idx

        return id2label, label2id

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, max_len=100)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        tokens=[]
        for i in input_ids:
            tokens_id = i.cpu().detach().numpy()
            text=self.tokenizer.convert_ids_to_kmers(tokens_id)
            tokens.append(text)
        return tokens

    @property
    def predicted_class_index(self) -> int:
        "Returns predicted class index (int) for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            preds = self.model(self.input_ids)
            self.pred_class = (torch.sigmoid(preds)>0.5).int()
            return self.pred_class.cpu().detach().numpy()


    @property
    def predicted_class_name(self):
        "Returns predicted class name (str) for model with last calculated `input_ids`"
        try:
            index = self.predicted_class_index
           # print(self.id2label)
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    @property
    def word_attributions(self) -> list:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions is not None:
            return self.attributions.word_attributions
        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")


    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
	
        preds = self.model(input_ids, segment_ids, attention_mask)

        # if it is a single output node
        self._single_node_output = True
        self.pred_probs = torch.sigmoid(preds)[0]
        return torch.sigmoid(preds)


    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.segment_ids,
        ) = self._make_input_reference_pair(self.text)

        (
            self.segment_ids,
            self.ref_segment_ids,
        ) = self._make_input_reference_segment_pair(self.segment_ids)
        
        self.attention_mask = self._make_attention_mask(self.input_ids)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            for cn in class_name:
                if int(torch.tensor(0).numpy()) in self.label2id.keys():
                    self.selected_index = int(self.label2id[int(torch.tensor(0).numpy())])
                    self.pred_index = self.predicted_class_index

        reference_tokens = [token for token in self.decode(self.input_ids)]
        lig = LIGAttributions(
            self._forward,
            embeddings,
            reference_tokens,
            self.input_ids,
            self.ref_input_ids,
            self.attention_mask,
            segment_ids=self.segment_ids,
            ref_segment_ids=self.ref_segment_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps
        )
        lig.summarize()
        self.attributions = lig

    def _run(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = None,
    ) -> list:  # type: ignore
        embeddings = self.word_embeddings

        self.text = self._clean_text(text)

        self._calculate_attributions(embeddings=embeddings, index=index, class_name=class_name)
        pred_class = self.pred_class.to("cpu").int()
        return self.word_attributions,pred_class  # type: ignore

    def __call__(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional): Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(text, index, class_name, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s
