import torch
import torch.nn as nn
import copy
import json
from model.layers import BertAttention, BertIntermediate, BertOutput, activations

def lm_mask(segment_ids):
    """?????Attention Mask(?????)
    """
    idxs = torch.arange(0, segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask


def unilm_mask(tokens_ids, segment_ids):
    """??UniLM?Attention Mask(Seq2Seq???)
        ??source?target???,?segment_ids????
        UniLM: https://arxiv.org/abs/1905.03197

        token_ids: shape?(batch_size, seq_length), type?tensor
        segment_ids: shape?(batch_size, seq_length), type?tensor
    """

    # ?segment_ids?padding?????1,????????padding,?????padding???mask???0
    ids = segment_ids + (tokens_ids <= 0).long()
    # ???????????
    idxs = torch.cumsum(ids, dim=1)
    # ??tokens_ids??mask??:[batch_size, 1, seq_length, 1]
    extended_mask = tokens_ids.unsqueeze(1).unsqueeze(3)
    # ??unilm?mask??,??shape???[batch_size, num_heads, from_seq_length, to_seq_length]
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).to(dtype=torch.float32)
    # ?padding???mask???0
    mask *= extended_mask
    return mask
    
class PreTrainedModel(nn.Module):

    def __init__(self):
        super().__init__()
       

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        # Prune heads if needed
        self.apply(self._init_weights)


####################################################################################
#       bert                                                                       #
####################################################################################

class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

class BertEmbeddings(nn.Module):
    """
        embeddings?
        ??word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position, segment_vocab_size, drop_rate):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=2)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size)

        self.layerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, token_ids, segment_ids=None):
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
        Transformer?:
        ???: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

        ??: 1??????dropout?,??????dropout,????dropout??????,????
              2????Transformer?encoder??Feed Forward??????linear,
              config.intermediate_size?????????linear?????,?????linear?????
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, intermediate_size, hidden_act, is_dropout=False):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout_rate)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size, dropout_rate)

    def forward(self, hidden_states, attention_mask):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """??BERT??
    """

    def __init__(
            self,
            hidden_size,  # ??????
            num_attention_heads,
            dropout_rate,
            intermediate_size,
            hidden_act,
            num_hidden_layers=1,
            is_dropout=False,
            initializer_range=0.02, # ???????
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range


        super().__init__()

        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.intermediate_size, self.hidden_act, is_dropout=False)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, mask):
        """
            token_ids: ???token?vocab????id
            segment_ids: ??token?????id,??0?1(0?????token?????,1???????),?
                             ???????????,segment_ids??????0,?????
            attention_mask:??????0?1,???padding?token???attention, 1??attetion, 0???attention

            ???????shape?: (batch_size, sequence_length); type?tensor
        """
        all_hidden_states = ()
        for i, layer_module in enumerate(self.encoderLayer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, mask)
            hidden_states = layer_outputs[0]
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states
                ]
                if v is not None
            )

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERTModel(BertPreTrainedModel):

    def __init__(self, encoder, vocab_size, hidden_size, max_position, segment_vocab_size,dropout_rate):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.dropout_rate = dropout_rate
        self.encoder = encoder
        self.embedding = BertEmbeddings(self.vocab_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate)
        self.pooler = BertPooler(hidden_size)
        self.classification_layer = nn.Linear(hidden_size, 1)

        self.post_init()

    def forward(self, token_ids, segment_ids=None, attention_mask=None):

        embedded_sources = self.embedding(token_ids, segment_ids)

        if attention_mask is None:
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        encoder_outputs = self.encoder(embedded_sources, attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        pooled_output = outputs[1]
        classification_output = self.classification_layer(pooled_output)
        return classification_output



def build_model(hidden_size, num_attention_heads, num_hidden_layers,dropout_rate, intermediate_size, hidden_act, vocab_size, max_position, segment_vocab_size):

    encoder = BertEncoder(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        num_hidden_layers=num_hidden_layers)

    bert = BERTModel(
        encoder=encoder,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position=max_position,
        segment_vocab_size=segment_vocab_size,
        dropout_rate=dropout_rate)

    return bert
