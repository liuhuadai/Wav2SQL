from copy import deepcopy
import itertools
import operator
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch import nn
from typing import List, Dict, Any, Optional, Union
# from dgmc.models.dgmc import DGMC
# from dgmc.models.gated_graph_conv import GatedGraphConv
from wav2sql.models import transformer
from wav2sql.models import variational_lstm
from wav2sql.utils import batched_sequence
import torch.nn.functional as F
import math
from torch import Tensor
# from wav2sql.sigma.graph_matching.gnns import GIN
# from wav2sql.sigma.graph_matching.model import SIGMA
# from research.adversarial_training.reversal_encoder import ReversalEncoder

INF = 100000000
NINF = -INF

def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def get_attn_mask(seq_lengths):
    # Given seq_lengths like [3, 1, 2], this will produce
    # [[[1, 1, 1],
    #   [1, 1, 1],
    #   [1, 1, 1]],
    #  [[1, 0, 0],
    #   [0, 0, 0],
    #   [0, 0, 0]],
    #  [[1, 1, 0],
    #   [1, 1, 0],
    #   [0, 0, 0]]]
    # int(max(...)) so that it has type 'int instead of numpy.int64
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask


class LookupEmbeddings(torch.nn.Module):
    def __init__(self, device, vocab, embedder, emb_size, learnable_words=[]):
        super().__init__()
        self._device = device
        self.vocab = vocab
        self.embedder = embedder
        self.emb_size = emb_size

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=emb_size)
        if self.embedder:
            assert emb_size == self.embedder.dim

        # init embedding
        self.learnable_words = learnable_words
        init_embed_list = []
        for i, word in enumerate(self.vocab):
            if self.embedder.contains(word):
                init_embed_list.append( \
                    self.embedder.lookup(word))
            else:
                init_embed_list.append(self.embedding.weight[i])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)

    def forward_unbatched(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        embs = []
        for tokens in token_lists:
            # token_indices shape: batch (=1) x length
            token_indices = torch.tensor(
                self.vocab.indices(tokens), device=self._device).unsqueeze(0)

            # emb shape: batch (=1) x length x word_emb_size
            emb = self.embedding(token_indices)

            # emb shape: desc length x batch (=1) x word_emb_size
            emb = emb.transpose(0, 1)
            embs.append(emb)

        # all_embs shape: sum of desc lengths x batch (=1) x word_emb_size
        all_embs = torch.cat(embs, dim=0)

        # boundaries shape: num of descs + 1
        # If desc lengths are [2, 3, 4],
        # then boundaries is [0, 2, 5, 9]
        boundaries = np.cumsum([0] + [emb.shape[0] for emb in embs])

        return all_embs, boundaries

    def _compute_boundaries(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists]

        return boundaries

    def _embed_token(self, token, batch_idx):
        if token in self.learnable_words or not self.embedder.contains(token):
            return self.embedding.weight[self.vocab.index(token)]
        else:
            emb = self.embedder.lookup(token)
            return emb.to(self._device)

    def forward(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.emb_size,),
            device=self._device,
            item_to_tensor=self._embed_token)
        all_embs = all_embs.apply(lambda d: d.to(self._device))
        boundaries = self._compute_boundaries(token_lists)
        return all_embs, boundaries

    def _embed_words_learned(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        # PackedSequencePlus, with shape: [batch, num descs * desc length (sum of desc lengths)]
        indices = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(1,),  # For compatibility with old PyTorch versions
            tensor_type=torch.LongTensor,
            item_to_tensor=lambda token, batch_idx, out: out.fill_(self.vocab.index(token))
        )
        indices = indices.apply(lambda d: d.to(self._device))
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = indices.apply(lambda x: self.embedding(x.squeeze(-1)))

        return all_embs, self._compute_boundaries(token_lists)


class UnitEmbeddings(torch.nn.Module):
    def __init__(self, device, vocab, emb_size, learnable_words=[]):
        super().__init__()
        self._device = device
        self.vocab = vocab
        # self.embedder = embedder
        self.emb_size = emb_size

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=emb_size)
        # if self.embedder:
        #     assert emb_size == self.embedder.dim

        # init embedding
        self.learnable_words = learnable_words
        init_embed_list = []
        for i, word in enumerate(self.vocab):
            init_embed_list.append(self.embedding.weight[i])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)

    def forward_unbatched(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        embs = []
        for tokens in token_lists:
            # token_indices shape: batch (=1) x length
            token_indices = torch.tensor(
                self.vocab.indices(tokens), device=self._device).unsqueeze(0)

            # emb shape: batch (=1) x length x word_emb_size
            emb = self.embedding(token_indices)

            # emb shape: desc length x batch (=1) x word_emb_size
            emb = emb.transpose(0, 1)
            embs.append(emb)

        # all_embs shape: sum of desc lengths x batch (=1) x word_emb_size
        all_embs = torch.cat(embs, dim=0)

        # boundaries shape: num of descs + 1
        # If desc lengths are [2, 3, 4],
        # then boundaries is [0, 2, 5, 9]
        boundaries = np.cumsum([0] + [emb.shape[0] for emb in embs])

        return all_embs, boundaries

    def _compute_boundaries(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists]

        return boundaries

    def _embed_token(self, token, batch_idx):
        # if token in self.learnable_words or not self.embedder.contains(token):
        #     return self.embedding.weight[self.vocab.index(token)]
        # else:
        #     emb = self.embedder.lookup(token)
        #     return emb.to(self._device)
        return self.embedding.weight[self.vocab.index(token)]

    def forward(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.emb_size,),
            device=self._device,
            item_to_tensor=self._embed_token)
        all_embs = all_embs.apply(lambda d: d.to(self._device))

        return all_embs, self._compute_boundaries(token_lists)

    def _embed_words_learned(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        # PackedSequencePlus, with shape: [batch, num descs * desc length (sum of desc lengths)]
        indices = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(1,),  # For compatibility with old PyTorch versions
            tensor_type=torch.LongTensor,
            item_to_tensor=lambda token, batch_idx, out: out.fill_(self.vocab.index(token))
        )
        indices = indices.apply(lambda d: d.to(self._device))
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = indices.apply(lambda x: self.embedding(x.squeeze(-1)))

        return all_embs, self._compute_boundaries(token_lists)


class EmbLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input_):
        all_embs, boundaries = input_
        all_embs = all_embs.apply(lambda d: self.linear(d))
        return all_embs, boundaries


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout, summarize, use_native=False):
        # input_size: dimensionality of input
        # output_size: dimensionality of output
        # dropout
        # summarize:
        # - True: return Tensor of 1 x batch x emb size 
        # - False: return Tensor of seq len x batch x emb size 
        super().__init__()
        
        if use_native:
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=output_size // 2,
                bidirectional=True,
                dropout=dropout)
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.lstm = variational_lstm.LSTM(
                input_size=input_size,
                hidden_size=int(output_size // 2),
                bidirectional=True,
                dropout=dropout)
        self.summarize = summarize
        self.use_native = use_native

    def forward_unbatched(self, input_):
        # all_embs shape: sum of desc lengths x batch (=1) x input_size
        all_embs, boundaries = input_

        new_boundaries = [0]
        outputs = []
        for left, right in zip(boundaries, boundaries[1:]):
            # state shape:
            # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # output shape: seq len x batch size x output_size
            if self.use_native:
                inp = self.dropout(all_embs[left:right])
                output, (h, c) = self.lstm(inp)
            else:
                output, (h, c) = self.lstm(all_embs[left:right])
            if self.summarize:
                seq_emb = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
                new_boundaries.append(new_boundaries[-1] + 1)
            else:
                seq_emb = output
                new_boundaries.append(new_boundaries[-1] + output.shape[0])
            outputs.append(seq_emb)

        return torch.cat(outputs, dim=0), new_boundaries

    def forward(self, input_):
        # all_embs shape: PackedSequencePlus with shape [batch, sum of desc lengths, input_size]
        # boundaries: list of lists with shape [batch, num descs + 1]
        all_embs, boundaries = input_

        # List of the following:
        # (batch_idx, desc_idx, length)
        desc_lengths = []
        batch_desc_to_flat_map = {}
        for batch_idx, boundaries_for_item in enumerate(boundaries):
            for desc_idx, (left, right) in enumerate(zip(boundaries_for_item, boundaries_for_item[1:])):
                desc_lengths.append((batch_idx, desc_idx, right - left))
                batch_desc_to_flat_map[batch_idx, desc_idx] = len(batch_desc_to_flat_map)

        # Recreate PackedSequencePlus into shape
        # [batch * num descs, desc length, input_size]
        # with name `rearranged_all_embs`
        remapped_ps_indices = []

        def rearranged_all_embs_map_index(desc_lengths_idx, seq_idx):
            batch_idx, desc_idx, _ = desc_lengths[desc_lengths_idx]
            return batch_idx, boundaries[batch_idx][desc_idx] + seq_idx

        def rearranged_all_embs_gather_from_indices(indices):
            batch_indices, seq_indices = zip(*indices)
            remapped_ps_indices[:] = all_embs.raw_index(batch_indices, seq_indices)
            return all_embs.ps.data[torch.LongTensor(remapped_ps_indices)]

        rearranged_all_embs = batched_sequence.PackedSequencePlus.from_gather(
            lengths=[length for _, _, length in desc_lengths],
            map_index=rearranged_all_embs_map_index,
            gather_from_indices=rearranged_all_embs_gather_from_indices)
        rev_remapped_ps_indices = tuple(
            x[0] for x in sorted(
                enumerate(remapped_ps_indices), key=operator.itemgetter(1)))

        # output shape: PackedSequence, [batch * num_descs, desc length, output_size]
        # state shape:
        # - h: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        # - c: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        if self.use_native:
            rearranged_all_embs = rearranged_all_embs.apply(self.dropout)
        output, (h, c) = self.lstm(rearranged_all_embs.ps)
        
        if self.summarize:
            # h shape: [batch * num descs, output_size]
            h = torch.cat((h[0], h[1]), dim=-1)

            # new_all_embs: PackedSequencePlus, [batch, num descs, input_size]
            new_all_embs = batched_sequence.PackedSequencePlus.from_gather(
                lengths=[len(boundaries_for_item) - 1 for boundaries_for_item in boundaries],
                map_index=lambda batch_idx, desc_idx: rearranged_all_embs.sort_to_orig[
                    batch_desc_to_flat_map[batch_idx, desc_idx]],
                gather_from_indices=lambda indices: h[torch.LongTensor(indices)])

            new_boundaries = [
                list(range(len(boundaries_for_item)))
                for boundaries_for_item in boundaries
            ]
        else:
            new_all_embs = all_embs.apply(
                lambda _: output.data[torch.LongTensor(rev_remapped_ps_indices)])
            new_boundaries = boundaries
        
        return new_all_embs, new_boundaries


class RelationalTransformerUpdate(torch.nn.Module):

    def __init__(self, device, num_layers, num_heads, hidden_size,
                 ff_size=None,
                 dropout=0.1,
                 merge_types=False,
                 tie_layers=False,
                 qq_max_dist=2,
                 # qc_token_match=True,
                 # qt_token_match=True,
                 # cq_token_match=True,
                 cc_foreign_key=True,
                 cc_table_match=True,
                 cc_max_dist=2,
                 ct_foreign_key=True,
                 ct_table_match=True,
                 # tq_token_match=True,
                 tc_table_match=True,
                 tc_foreign_key=True,
                 tt_max_dist=2,
                 tt_foreign_key=True,
                 sc_link=False,
                 cv_link=False,
                 co_attn=False,
                 train_sc=False,
                 sadga_dropout=0.5,
                 activation_func='tanh',
                 ggnn_num_layers=2,
                 num_steps=10,
                 coattn_fusion=False
                 ):
        super().__init__()
        self._device = device
        self.num_heads = num_heads
        self.coattn_fusion = coattn_fusion
        self.qq_max_dist = qq_max_dist
        # self.qc_token_match = qc_token_match
        # self.qt_token_match = qt_token_match
        # self.cq_token_match = cq_token_match
        self.cc_foreign_key = cc_foreign_key
        self.cc_table_match = cc_table_match
        self.cc_max_dist = cc_max_dist
        self.ct_foreign_key = ct_foreign_key
        self.ct_table_match = ct_table_match
        # self.tq_token_match = tq_token_match
        self.tc_table_match = tc_table_match
        self.tc_foreign_key = tc_foreign_key
        self.tt_max_dist = tt_max_dist
        self.tt_foreign_key = tt_foreign_key

        self.relation_ids = {}

        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)

        def add_rel_dist(name, max_dist):
            for i in range(-max_dist, max_dist + 1):
                add_relation((name, i))

        add_rel_dist('qq_dist', qq_max_dist)

        add_relation('qc_default')
        # if qc_token_match:
        #    add_relation('qc_token_match')

        add_relation('qt_default')
        # if qt_token_match:
        #    add_relation('qt_token_match')

        add_relation('cq_default')
        # if cq_token_match:
        #    add_relation('cq_token_match')

        add_relation('cc_default')
        if cc_foreign_key:
            add_relation('cc_foreign_key_forward')
            add_relation('cc_foreign_key_backward')
        if cc_table_match:
            add_relation('cc_table_match')
        add_rel_dist('cc_dist', cc_max_dist)

        add_relation('ct_default')
        if ct_foreign_key:
            add_relation('ct_foreign_key')
        if ct_table_match:
            add_relation('ct_primary_key')
            add_relation('ct_table_match')
            add_relation('ct_any_table')

        add_relation('tq_default')
        # if cq_token_match:
        #    add_relation('tq_token_match')

        add_relation('tc_default')
        if tc_table_match:
            add_relation('tc_primary_key')
            add_relation('tc_table_match')
            add_relation('tc_any_table')
        if tc_foreign_key:
            add_relation('tc_foreign_key')

        add_relation('tt_default')
        if tt_foreign_key:
            add_relation('tt_foreign_key_forward')
            add_relation('tt_foreign_key_backward')
            add_relation('tt_foreign_key_both')
        add_rel_dist('tt_dist', tt_max_dist)

        # schema linking relations
        # forward_backward
        if sc_link:
            add_relation('qcCEM')
            add_relation('cqCEM')
            add_relation('qtTEM')
            add_relation('tqTEM')
            add_relation('qcCPM')
            add_relation('cqCPM')
            add_relation('qtTPM')
            add_relation('tqTPM')

        if cv_link:
            add_relation("qcNUMBER")
            add_relation("cqNUMBER")
            add_relation("qcTIME")
            add_relation("cqTIME")
            add_relation("qcCELLMATCH")
            add_relation("cqCELLMATCH")

        if merge_types:
            assert not cc_foreign_key
            assert not cc_table_match
            assert not ct_foreign_key
            assert not ct_table_match
            assert not tc_foreign_key
            assert not tc_table_match
            assert not tt_foreign_key

            assert cc_max_dist == qq_max_dist
            assert tt_max_dist == qq_max_dist

            add_relation('xx_default')
            self.relation_ids['qc_default'] = self.relation_ids['xx_default']
            self.relation_ids['qt_default'] = self.relation_ids['xx_default']
            self.relation_ids['cq_default'] = self.relation_ids['xx_default']
            self.relation_ids['cc_default'] = self.relation_ids['xx_default']
            self.relation_ids['ct_default'] = self.relation_ids['xx_default']
            self.relation_ids['tq_default'] = self.relation_ids['xx_default']
            self.relation_ids['tc_default'] = self.relation_ids['xx_default']
            self.relation_ids['tt_default'] = self.relation_ids['xx_default']

            if sc_link:
                self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
                self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
            if cv_link:
                self.relation_ids["qcNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["cqNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["qcTIME"] = self.relation_ids['xx_default']
                self.relation_ids["cqTIME"] = self.relation_ids['xx_default']
                self.relation_ids["qcCELLMATCH"] = self.relation_ids['xx_default']
                self.relation_ids["cqCELLMATCH"] = self.relation_ids['xx_default']

            for i in range(-qq_max_dist, qq_max_dist + 1):
                self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist', i]
                self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist', i]

        if ff_size is None:
            ff_size = hidden_size * 4
        
        # self.sc_link_scorer = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, hidden_size)
        # )
        # # self.cv_link_scorer = nn.Sequential(
        # #     nn.Linear(hidden_size*2, hidden_size),
        # #     nn.Tanh(),
        # #     nn.Linear(hidden_size, 1)
        # # )
        # self.q_share = GatedEmbeddingUnit(hidden_size, hidden_size)
        # self.sc_share = GatedEmbeddingUnit(hidden_size, hidden_size)
        
        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout),
                transformer.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                len(self.relation_ids),
                dropout,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout) if co_attn else None
                    ),
            hidden_size,
            num_layers,
            tie_layers)

        self.align_attn = transformer.PointerWithRelations(hidden_size,
                                                           len(self.relation_ids), dropout)

    def create_align_mask(self, num_head, q_length, c_length, t_length):
        # mask with size num_heads * all_len * all * len
        all_length = q_length + c_length + t_length
        mask_1 = torch.ones(num_head - 1, all_length, all_length, device=self._device)
        mask_2 = torch.zeros(1, all_length, all_length, device=self._device)
        for i in range(q_length):
            for j in range(q_length, q_length + c_length):
                mask_2[0, i, j] = 1
                mask_2[0, j, i] = 1
        mask = torch.cat([mask_1, mask_2], 0)
        return mask

    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries,step=0):
        loss = 0.0
        # enc shape: batch (=1) x total len x recurrent size

        # schema = torch.cat((c_enc, t_enc), dim=0).transpose(0,1)
        # q_t = q_enc.transpose(0,1)
        # # schema-aware
        # q_sche_aware, _ = transformer.attention(q_t, schema, schema)
        # q_sche = self.fusion_module(q_t, q_sche_aware)
        # # q_sche = q_enc + q_sche_aware
        # sche_q_aware, _ = transformer.attention(schema, q_t, q_t)
        # sche_q = self.fusion_module(schema, sche_q_aware)
        # # sche_q = schema+sche_q_aware
        # enc = torch.cat((q_sche, sche_q), dim=1)
        enc = torch.cat((q_enc, c_enc, t_enc), dim=0)
        enc = enc.transpose(0, 1)
        # Catalogue which things are where
        relations = self.compute_relations(
        desc,
        enc_length=enc.shape[1],
        q_enc_length=q_enc.shape[0],
        c_enc_length=c_enc.shape[0],
        c_boundaries=c_boundaries,
        t_boundaries=t_boundaries)

        relations_t = torch.LongTensor(relations).to(self._device)
        
        enc_new = self.encoder(enc, relations_t, mask=None, q_len=q_enc.shape[0] )
        # Split updated_enc again
        c_base = q_enc.shape[0]
        t_base = q_enc.shape[0] + c_enc.shape[0]
        q_enc_new = enc_new[:, :c_base]
        c_enc_new = enc_new[:, c_base:t_base]
        t_enc_new = enc_new[:, t_base:]

        # #### use contrastive loss
        # q_n = q_enc_new.squeeze(0)
        # sc_n = torch.cat((c_enc_new, t_enc_new), dim=1).squeeze(0)
        # # q_norm = normalize_embeddings(self.q_share(q_n))
        # # sc_norm = normalize_embeddings(self.sc_share(sc_n))
        # ########
        # loss = NormSoftmaxLoss()(sim_matrix(q_n, sc_n))
        # loss = NormSoftmaxLoss()(sim_matrix(q_norm, sc_norm))
        # q_enc_new = q_norm.unsqueeze(0)
        # c_enc_new = sc_norm[:c_enc.shape[0]].unsqueeze(0)
        # t_enc_new = sc_norm[c_enc.shape[0]:].unsqueeze(0)
        # loss = NormSoftmaxLoss()(sim_matrix(q_n, sc_n))
        m2c_align_mat = self.align_attn(enc_new, enc_new[:, c_base:t_base], \
                                        enc_new[:, c_base:t_base], relations_t[:, c_base:t_base])
        m2t_align_mat = self.align_attn(enc_new, enc_new[:, t_base:], \
                                        enc_new[:, t_base:], relations_t[:, t_base:])
        return q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat), loss

    def forward(self, descs, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
        # TODO: Update to also compute m2c_align_mat and m2t_align_mat
        # enc: PackedSequencePlus with shape [batch, total len, recurrent size]
        enc = batched_sequence.PackedSequencePlus.cat_seqs((q_enc, c_enc, t_enc))

        q_enc_lengths = list(q_enc.orig_lengths())
        c_enc_lengths = list(c_enc.orig_lengths())
        t_enc_lengths = list(t_enc.orig_lengths())
        enc_lengths = list(enc.orig_lengths())
        max_enc_length = max(enc_lengths)

        all_relations = []
        for batch_idx, desc in enumerate(descs):
            enc_length = enc_lengths[batch_idx]
            relations_for_item = self.compute_relations(
                desc,
                enc_length,
                q_enc_lengths[batch_idx],
                c_enc_lengths[batch_idx],
                c_boundaries[batch_idx],
                t_boundaries[batch_idx])
            all_relations.append(np.pad(relations_for_item, ((0, max_enc_length - enc_length),), 'constant'))
        relations_t = torch.from_numpy(np.stack(all_relations)).to(self._device)

        # mask shape: [batch, total len, total len]
        mask = get_attn_mask(enc_lengths).to(self._device)
        # enc_new: shape [batch, total len, recurrent size]
        enc_padded, _ = enc.pad(batch_first=True)
        enc_new = self.encoder(enc_padded, relations_t, mask=mask)

        # Split enc_new again
        def gather_from_enc_new(indices):
            batch_indices, seq_indices = zip(*indices)
            return enc_new[torch.LongTensor(batch_indices), torch.LongTensor(seq_indices)]

        q_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=q_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (batch_idx, seq_idx),
            gather_from_indices=gather_from_enc_new)
        c_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=c_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (batch_idx, q_enc_lengths[batch_idx] + seq_idx),
            gather_from_indices=gather_from_enc_new)
        t_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=t_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (
            batch_idx, q_enc_lengths[batch_idx] + c_enc_lengths[batch_idx] + seq_idx),
            gather_from_indices=gather_from_enc_new)
        return q_enc_new, c_enc_new, t_enc_new

    def compute_relations(self, desc, enc_length, q_enc_length, c_enc_length, c_boundaries, t_boundaries):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})
        # sc_link = {'q_col_match': {}, 'q_tab_match': {}}
        # cv_link = {'num_date_match': {}, 'cell_match': {}}
        # i = 0
        # sc_link_tmp = deepcopy(sc_link)
        # cv_link_tmp = deepcopy(cv_link)
        # for link in sc_link_tmp.keys():
        #     for ql in sc_link_tmp[link].keys():
        #         ql_n = ql.split(',')
        #         x,y = ql_n[0], ql_n[1]
        #         x = str(i)
        #         i+=1
        #         sc_link[link][f'{x},{y}'] = sc_link_tmp[link][ql]
        #         sc_link[link].pop(ql)
        # for link in cv_link_tmp.keys():
        #     for ql in cv_link_tmp[link].keys():
        #         ql_n = ql.split(',')
        #         x,y = ql_n[0], ql_n[1]
        #         x = str(i)
        #         i+=1
        #         cv_link[link][f'{x},{y}'] = cv_link_tmp[link][ql]
        #         cv_link[link].pop(ql)

        # Catalogue which things are where
        loc_types = {}
        for i in range(q_enc_length):
            loc_types[i] = ('question',)

        c_base = q_enc_length
        for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
            for i in range(c_start + c_base, c_end + c_base):
                loc_types[i] = ('column', c_id)
        t_base = q_enc_length + c_enc_length
        for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
            for i in range(t_start + t_base, t_end + t_base):
                loc_types[i] = ('table', t_id)

        relations = np.empty((enc_length, enc_length), dtype=np.int64)
        
        for i, j in itertools.product(range(enc_length), repeat=2):
            def set_relation(name):
                relations[i, j] = self.relation_ids[name]

            i_type, j_type = loc_types[i], loc_types[j]
            if i_type[0] == 'question':
                if j_type[0] == 'question':
                    set_relation(('qq_dist', clamp(j - i, self.qq_max_dist)))
                elif j_type[0] == 'column':
                    # set_relation('qc_default')
                    j_real = j - c_base
                    if f"{i},{j_real}" in sc_link["q_col_match"]:
                        set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["cell_match"]:
                        set_relation("qc" + cv_link["cell_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["num_date_match"]:
                        set_relation("qc" + cv_link["num_date_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qc_default')
                elif j_type[0] == 'table':
                    # set_relation('qt_default')
                    j_real = j - t_base
                    if f"{i},{j_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qt_default')

            elif i_type[0] == 'column':
                if j_type[0] == 'question':
                    # set_relation('cq_default')
                    i_real = i - c_base
                    if f"{j},{i_real}" in sc_link["q_col_match"]:
                        set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["cell_match"]:
                        set_relation("cq" + cv_link["cell_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["num_date_match"]:
                        set_relation("cq" + cv_link["num_date_match"][f"{j},{i_real}"])
                    else:
                        set_relation('cq_default')
                elif j_type[0] == 'column':
                    col1, col2 = i_type[1], j_type[1]
                    if col1 == col2:
                        set_relation(('cc_dist', clamp(j - i, self.cc_max_dist)))
                    else:
                        set_relation('cc_default')
                        if self.cc_foreign_key:
                            if desc['foreign_keys'].get(str(col1)) == col2:
                                set_relation('cc_foreign_key_forward')
                            if desc['foreign_keys'].get(str(col2)) == col1:
                                set_relation('cc_foreign_key_backward')
                        if (self.cc_table_match and
                                desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
                            set_relation('cc_table_match')

                elif j_type[0] == 'table':
                    col, table = i_type[1], j_type[1]
                    set_relation('ct_default')
                    if self.ct_foreign_key and self.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    if self.ct_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('ct_primary_key')
                            else:
                                set_relation('ct_table_match')
                        elif col_table is None:
                            set_relation('ct_any_table')

            elif i_type[0] == 'table':
                if j_type[0] == 'question':
                    # set_relation('tq_default')
                    i_real = i - t_base
                    if f"{j},{i_real}" in sc_link["q_tab_match"]:
                        set_relation("tq" + sc_link["q_tab_match"][f"{j},{i_real}"])
                    else:
                        set_relation('tq_default')
                elif j_type[0] == 'column':
                    table, col = i_type[1], j_type[1]
                    set_relation('tc_default')

                    if self.tc_foreign_key and self.match_foreign_key(desc, col, table):
                        set_relation('tc_foreign_key')
                    if self.tc_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('tc_primary_key')
                            else:
                                set_relation('tc_table_match')
                        elif col_table is None:
                            set_relation('tc_any_table')
                elif j_type[0] == 'table':
                    table1, table2 = i_type[1], j_type[1]
                    if table1 == table2:
                        set_relation(('tt_dist', clamp(j - i, self.tt_max_dist)))
                    else:
                        set_relation('tt_default')
                        if self.tt_foreign_key:
                            forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                            backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                            if forward and backward:
                                set_relation('tt_foreign_key_both')
                            elif forward:
                                set_relation('tt_foreign_key_forward')
                            elif backward:
                                set_relation('tt_foreign_key_backward')
        return relations
        
    def get_links(self, desc, q_length, col_num, tbl_num):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})
        sc_rels = torch.full((q_length,), col_num + tbl_num)
        cv_rels = torch.full((q_length,), col_num + tbl_num)
        for k1, v1 in sc_link.items():
            for k2, v2 in v1.items():
                if k1=='q_col_match':
                    l = k2.split(',')
                    i, j = int(l[0]), int(l[1])
                    sc_rels[i] = j
                    cv_rels[i] = -100
                else:
                    l = k2.split(',')
                    i, j = int(l[0]), int(l[1])
                    sc_rels[i] = j + col_num
                    # assert sc_rels[i] <= col_num+tbl_num, j
                    cv_rels[i] = -100
        for k1, v1 in cv_link.items():
            for k2, v2 in v1.items():
                l = k2.split(',')
                i, j = int(l[0]), int(l[1])
                cv_rels[i] = j
        return sc_rels, cv_rels

    

    def _add_relation_node(self, x, relation):
        num_item = x.size(1)
        relation_index = num_item

        adj_forward = []
        adj_back = []
        adj_self = []

        relation_ids = []
        for i, j in itertools.product(range(num_item), repeat=2):
            if i <= j:
                continue
            relation_id = relation[i][j]
            if relation_id != 0:
                relation_ids.append(relation_id)
                adj_forward.append((i, j))
                adj_forward.append((j, i))
                # adj_back.append((j, relation_index))
                # adj_back.append((relation_index, i))
                relation_index = relation_index + 1
        num_node = relation_index
        for i in range(num_node):
            adj_self.append((i, i))

        # relations_t = torch.LongTensor(relation_ids).to(self._device)
        # relation_emb = self.rel_v_emb(relations_t)
        input = x.squeeze(0)
        # input = torch.cat([x.squeeze(0), relation_emb], dim=0)
        all_adj_types = [adj_forward, adj_self]

        adj_list = []
        for i in range(len(all_adj_types)):
            adj_t = input.new_zeros((input.size(0), input.size(0)))
            adj_tuple = all_adj_types[i]
            for edge in adj_tuple:
                adj_t[edge[1], edge[0]] = 1
            adj_list.append(adj_t.to_sparse())

        return input, adj_list

    
    def reward_general(self, n1, n2, e1, e2, theta_in, miss_match_value=-1.0):

        theta = theta_in
        r_sum = 0
        for e in range(len(e1)):
            e1_dense = e1[e].to_dense()
            e2_dense = e2[e].to_dense()
            r = e1_dense @ theta @ e2_dense.transpose(-1, -2) * theta
            r = r.sum([0,1])
            r_sum += r
        
        return r_sum
            

        if miss_match_value > 0.0:
            ones_1 = torch.ones([1, n1, 1]).to(theta_in.device)
            ones_2 = torch.ones([1, n2, 1]).to(theta_in.device)

            r_mis_1 = e1_dense @ ones_1 @ ones_2.transpose(-1, -2) * theta
            r_mis_1 = r_mis_1.sum([1, 2]).mean()

            r_mis_2 = ones_1 @ ones_2.transpose(-1, -2) @ e2_dense.transpose(-1, -2) * theta
            r_mis_2 = r_mis_2.sum([1, 2]).mean()

            r_mis = r_mis_1 + r_mis_2 - 2.0*r

            r_mis = r_mis * miss_match_value

            r = r - r_mis

        return r

    @classmethod
    def match_foreign_key(cls, desc, col, table):
        foreign_key_for = desc['foreign_keys'].get(str(col))
        if foreign_key_for is None:
            return False

        foreign_table = desc['column_to_table'][str(foreign_key_for)]
        return desc['column_to_table'][str(col)] == foreign_table


class NoOpUpdate:
    def __init__(self, device, hidden_size):
        pass

    def __call__(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
        # return q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc.transpose(0, 1)
        return q_enc, c_enc, t_enc

    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
        """
        The same interface with RAT
        return: encodings with size: length * embed_size, alignment matrix
        """
        return q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc.transpose(0, 1), (None, None)


class FairseqDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x

class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m

class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class LSTMs(nn.Module):
    def __init__(self,
                input_size=1024, 
                output_size=512, 
                dropout=0.1
                 ):
        super().__init__()
        self.bidirectional = True 
        
        self.q_lstm = nn.LSTM(
                input_size=768,
                hidden_size=output_size // 2,
                bidirectional=True,
                dropout=dropout)
        self.c_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=output_size // 2,
                bidirectional=True,
                dropout=dropout)
        self.t_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=output_size // 2,
                bidirectional=True,
                dropout=dropout)
        # self.dropout = torch.nn.Dropout(dropout)
        # # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
    

    def forward(self, q, c, t):
        # print(x.shape)
        # x = self.dropout(x)
        q_enc, _ = self.q_lstm(q)
        c_enc, _ = self.c_lstm(c)
        t_enc, _ = self.t_lstm(t)
        # x = x.transpose(1,2).contiguous().unsqueeze(1)
        # output_lengths = self.get_seq_lens(lengths)
        # x, _ = self.conv(x, output_lengths)

       
        return q_enc, c_enc, t_enc
        # return output, hn

class ASR_Encoder(nn.Module):
    def __init__(self,
                #  input_size, output_size, dropout
                hidden_size=256,
                hidden_layers=3
                 ):
        super().__init__()
        self.bidirectional = True 

        # self.lstm = torch.nn.LSTM(
        #         input_size=input_size,
        #         hidden_size=output_size // 2,
        #         bidirectional=True,
        #         dropout=dropout)
        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv = MaskConv(nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True)
        # ))
        # # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = 768

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                rnn_type=nn.LSTM,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    rnn_type=nn.LSTM,
                    bidirectional=self.bidirectional
                ) for x in range(hidden_layers - 1)
            )
        )
        # fully_connected = nn.Sequential(
        #     nn.BatchNorm1d(hidden_size),
        #     nn.Linear(hidden_size, output_size, bias=False)
        # )
        # self.fc = nn.Sequential(
        #     SequenceWise(fully_connected),
        # )

    def forward(self, x, lengths):
        # print(x.shape)
        # x = self.dropout(x)
        # output, (hn, cn) = self.lstm(x)
        # x = x.transpose(1,2).contiguous().unsqueeze(1)
        lengths = lengths.cpu().int()
        # output_lengths = self.get_seq_lens(lengths)
        # x, _ = self.conv(x, output_lengths)

        # sizes = x.size()
        # x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        # x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, lengths)

        # x = self.fc(x)
        return x, lengths
        # return output, hn


    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()