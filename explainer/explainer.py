import inspect
import re
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple, Union

import torch
from model.tokenizer import _is_control, _is_whitespace


class BaseExplainer(ABC):
    def __init__(
        self,
        model,
        tokenizer,
        device,
        pretrain=False,
    ):
        self.pretrain = pretrain
        self.model = model
        
        self.tokenizer = tokenizer

        self.ref_token_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        self.sep_token_id = self.tokenizer.vocab[self.tokenizer.sep_token]

        self.cls_token_id = self.tokenizer.vocab[self.tokenizer.cls_token]
        
        self.accepts_position_ids = True

        self.accepts_segment_ids = True

        self.device = device

        if self.pretrain:
            self.word_embeddings = self.model.pretrained_model.embedding.word_embeddings
        else:
            self.word_embeddings = self.model.embedding.word_embeddings
        self.position_embeddings = None
        self.token_type_embeddings = None

        self._set_available_embedding_types()

    @abstractmethod
    def encode(self, text: str = None):
        """
        Encode given text with a model's tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, input_ids: torch.Tensor) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.


        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.

        """
        raise NotImplementedError

    @abstractproperty
    def word_attributions(self):
        raise NotImplementedError

    @abstractmethod
    def _run(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def _forward(self):
        """
        Forward defines a function for passing inputs
        through a models's forward method.

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_attributions(self):
        """
        Internal method for calculating the attribution
        values for the input text.

        """
        raise NotImplementedError

    def _make_input_reference_pair(self, text: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenizes `text` to numerical token id  representation `input_ids`,
        as well as creating another reference tensor `ref_input_ids` of the same length
        that will be used as baseline for attributions. Additionally
        the length of text without special tokens appended is prepended is also
        returned.

        Args:
            text (str): Text for which we are creating both input ids
            and their corresponding reference ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]
        """
        inputs_ids = []
        segments_ids = []
        for t in text:
            text_len = len(t)-2
            input_ids, segment_ids = self.tokenizer.encode(t, max_len=100)
            input_ids = input_ids + (100 - len(input_ids)) * [0]
            segment_ids = segment_ids + (100 - len(segment_ids)) * [0]
            inputs_ids.append(input_ids)
            segments_ids.append(segment_ids)
            ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * text_len + [self.sep_token_id]
        
        # if no special tokens were added
        
        return (
            torch.tensor(inputs_ids, device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            segments_ids,
        )

    def _make_input_reference_segment_pair(
        self, segment_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two tensors indicating the corresponding token types for the `input_ids`
        and a corresponding all zero reference token type tensor.
        Args:
            input_ids (torch.Tensor): Tensor of text converted to `input_ids`
            sep_idx (int, optional):  Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        
        seq_len = len(segment_ids[0])
        ref_segment_ids = seq_len * [0]

        return (
            torch.tensor(segment_ids, device=self.device),
            torch.tensor([ref_segment_ids], device=self.device),
        )

    def _make_input_reference_position_id_pair(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors for positional encoding of tokens for input_ids and zeroed tensor for reference ids.

        Args:
            input_ids (torch.Tensor): inputs to create positional encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = len(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids[0])

    def _clean_text(self, text: str) -> str:
        outputs = []
        for t in text:
            output = []
            for char in t:
                cp = ord(char)
                if cp == 0 or cp == 0xfffd or _is_control(char):
                    continue
                if _is_whitespace(char):
                    output.append(" ")
                else:
                    output.append(char)
            outputs.append("".join(output))
        return outputs

    def _model_forward_signature_accepts_parameter(self, parameter: str) -> bool:
        signature = inspect.signature(self.model.forward)
        parameters = signature.parameters
        return parameter in parameters

    def _set_available_embedding_types(self):
        if self.pretrain:
            self.model_embeddings = self.model.pretrained_model.embedding
            self.position_embeddings = self.model.pretrained_model.embedding.position_embeddings
            self.segment_embeddings = self.model.pretrained_model.embedding.segment_embeddings
        else:
            self.model_embeddings = self.model.embedding
            self.position_embeddings = self.model.embedding.position_embeddings
            self.segment_embeddings = self.model.embedding.segment_embeddings

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__}"
        s += ")"

        return s
