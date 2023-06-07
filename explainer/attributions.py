from typing import Callable, Tuple, List, Union, Optional

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_segment_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids.cuda()
        self.ref_input_ids = ref_input_ids.cuda()
        self.attention_mask = attention_mask.cuda()
        self.target = target
        self.segment_ids = segment_ids.cuda()
        self.position_ids = position_ids
        self.ref_segment_ids = ref_segment_ids.cuda()
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)
        self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.segment_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_segment_ids
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

    @property
    def word_attributions(self) -> list:
        was = []
        for ix,(attr,tokens) in enumerate(zip(self.attributions_sum,self.tokens)):
            wa = []
            if len(attr) >= 1:
                for i, (attribution,t) in enumerate(zip(attr,tokens)):
                    wa.append((t, float(attribution.cpu().data.numpy())))
            was.append(wa)
        return was


    def summarize(self, end_idx=None):
        attribution_sum = []
        for i in range(self._attributions.size(0)):
            attrs=self._attributions[i,:,:]
            attrs=attrs.sum(dim=-1).squeeze(0)
            attrs=attrs[:end_idx] / torch.norm(attrs[:end_idx])
            attribution_sum.append(attrs)
        self.attributions_sum = attribution_sum
