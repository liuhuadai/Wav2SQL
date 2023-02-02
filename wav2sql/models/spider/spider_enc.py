import collections
import itertools
import json
import os
import pickle
import attr
import numpy as np
import torch
import torchaudio
from transformers import BertModel, BertTokenizer
from munch import DefaultMunch
from wav2sql.models import abstract_preproc
from wav2sql.models.spider import spider_enc_modules
from wav2sql.models.spider.data_loader import load_audio, load_randomly_augmented_audio
from wav2sql.models.spider.spider_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking
)
from wav2sql.resources import corenlp
from wav2sql.utils import registry
from wav2sql.utils import serialization
from wav2sql.utils import vocab
# from fairseq.models.roberta import RobertaModel
# from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerEncoder
import speechpy
import scipy.io.wavfile as wav
import librosa

from research.adversarial_training.reversal_encoder import ReversalEncoder

@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,
                               include_table_name_in_column,
                               fix_issue_16_primary_keys,
                               bert=False):
    """If it's bert, we also cache the normalized version of 
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    if bert: assert not include_table_name_in_column

    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'
        if bert:
            # for bert, we take the representation of the first word
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(Bertokens(col_toks))
        else:
            column_name = [type_tok] + col_toks

        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1

    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bert:
            r.normalized_table_names.append(Bertokens(table_toks))
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r


class SpiderEncoderV2Preproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000,
            include_table_name_in_column=True,
            word_emb=None,
            count_tokens_in_word_emb_for_vocab=False,
            fix_issue_16_primary_keys=False,
            compute_sc_link=False,
            compute_cv_link=False,
            use_unit=False,
            audio_section='maspider-dg',
            db_path=None):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = registry.construct('word_emb', word_emb)
        self.normalize = True
        self.audio_section = audio_section
        self.data_dir = os.path.join(save_path, 'enc')
        self.include_table_name_in_column = include_table_name_in_column
        self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.texts = collections.defaultdict(list)
        self.db_path = db_path

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.unit_vocab_path = os.path.join(save_path, 'unit_vocab.json')
        self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        self.vocab_word_freq_path = os.path.join(save_path, 'enc_word_freq.json')
        self.vocab = None
        self.counted_db_ids = set()
        self.preprocessed_schemas = {}
        self.use_unit = use_unit

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

        if section == 'train':
            if item.schema.db_id in self.counted_db_ids:
                to_count = preprocessed['question']
            else:
                self.counted_db_ids.add(item.schema.db_id)
                to_count = itertools.chain(
                    preprocessed['question'],
                    *preprocessed['columns'],
                    *preprocessed['tables'])

            for token in to_count:
                count_token = (
                        self.word_emb is None or
                        self.count_tokens_in_word_emb_for_vocab or
                        self.word_emb.lookup(token) is None)
                if count_token:
                    self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, validation_info):
        question, question_for_copying = self._tokenize_for_copying(item.text, item.orig['question'])
        preproc_schema = self._preprocess_schema(item.schema)
        if self.compute_sc_link:
            assert preproc_schema.column_names[0][0].startswith("<type:")
            column_names_without_types = [col[1:] for col in preproc_schema.column_names]
            sc_link = compute_schema_linking(question, column_names_without_types, preproc_schema.table_names)
        else:
            sc_link = {"q_col_match": {}, "q_tab_match": {}}

        if self.compute_cv_link:
            cv_link = compute_cell_value_linking(question, item.schema)
        else:
            cv_link = {"num_date_match": {}, "cell_match": {}}

        return {
            'raw_question': item.orig['question'],
            'question': question,
            'question_for_copying': question_for_copying,
            'db_id': item.schema.db_id,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
        }

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column, self.fix_issue_16_primary_keys)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in vocab")
        self.vocab.save(self.vocab_path)
        self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.unit_vocab = vocab.Vocab.load(self.unit_vocab_path)
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)

    def parse_audio(self, audio_path):
        # if self.aug_conf and self.aug_conf.speed_volume_perturb:
        #     y = load_randomly_augmented_audio()(audio_path, self.sample_rate)
        # else:
        y = load_audio(audio_path)
        # if self.noise_injector:
        #     add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
        #     if add_noise:
        #         y = self.noise_injector.inject_noise(y)
        n_fft = int(16000 * 0.02)
        win_length = n_fft
        hop_length = int(16000 * 0.01)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window='hamming')
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        # if self.aug_conf and self.aug_conf.spec_augment:
        #     spect = spec_augment(spect)

        return spect



    def dataset(self, section):
        if self.use_unit:
            return [
                json.loads(line)
                for line in open(os.path.join(self.data_dir, section + '.jsonl'))]
        else:
            data = []
            # audio = np.load(os.path.join('/home1/liuhuadai/Transfer/huangrongjie1/projects/wav2sql/data', section+'_0_1.npy'))
            # # audio = np.load(f'spider/hubert/{section}_0_1.npy')
            # with open(os.path.join('/home1/liuhuadai/Transfer/huangrongjie1/projects/wav2sql/data', section+'_0_1.len')) as f:
            #     lengths = f.readlines()
            # sum = 0 
            if 'aug_train' in section:
                audio_section = 'aug_train'
            elif 'train' in section:
                audio_section = 'train'
            else:
                audio_section = 'dev'
            # print(audio_section)
            f_r = open(os.path.join(f'/apdcephfs/share_1316500/nlphuang/huangrongjie1/wav2sql/processed_feature/{self.audio_section}/',f'{audio_section}.bin'),'rb')            
            # f_r = open(os.path.join('/apdcephfs/share_1316500/nlphuang/huangrongjie1/wav2sql/processed_feature',section+'.bin'),'rb')
            questions = pickle.load(f_r)
            for i, (line,question) in enumerate(zip(open(os.path.join(self.data_dir, section + '.jsonl')),questions)):
                # length = int(lengths[i].strip())
                input = json.loads(line)
                # input['tables'].append(['none'])
                # if len(input['columns']) > 100:
                #     sum+=length
                #     continue
                input['question'] = question
                # sum+=length
                data.append(input)
            # else:
            #     for i, line in enumerate(open(os.path.join(self.data_dir, section + '_skip.jsonl'))):
            #         input = json.loads(line)
            #         # if len(input['columns']) > 100:
            #         #     continue
            #         data.append(input)
            # # print('In total, process %d samples , skip %d extremely large databases.' % (len(data), i - len(data)))
            return data

@registry.register('encoder', 'spiderv2')
class SpiderEncoderV2(torch.nn.Module):
    batched = True
    Preproc = SpiderEncoderV2Preproc

    def __init__(
            self,
            device,
            preproc,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.,
            question_encoder=('emb', 'bilstm'),
            column_encoder=('emb', 'bilstm'),
            table_encoder=('emb', 'bilstm'),
            update_config={},
            include_in_memory=('question', 'column', 'table'),
            batch_encs_update=True,
            top_k_learnable=0,
            use_unit=False,
            reversal_lambda=0.1,
            use_reverse=False):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.unit_vocab = preproc.unit_vocab
        self.vocab = preproc.vocab
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0
        word_freq = self.preproc.vocab_builder.word_freq
        top_k_words = set([_a[0] for _a in word_freq.most_common(top_k_learnable)])
        self.learnable_words = top_k_words

        self.include_in_memory = set(include_in_memory)
        self.dropout = dropout
        args = {
            'encoder_freezing_updates': 0,
            'conv_kernel_sizes': '5,5',
            'conv_channels': 1024,
            'encoder_layers': 12,
            'encoder_normalize_before': True,
            'encoder_embed_dim': 256,
            'encoder_ffn_embed_dim': 256*8,
            'encoder_attention_heads': 4,
            'dropout': 0.1,
            'input_feat_per_channel':80,
            'input_channels':1,
            'max_source_positions':6000,
            'ctc_weight':0.0
        }
        self.args = DefaultMunch.fromDict(args)
        self.use_unit = use_unit
        self.reversal_lambda = reversal_lambda
        if use_unit:
            self.question_encoder = self._build_modules(question_encoder)
        else:
            self.question_encoder = spider_enc_modules.ASR_Encoder(
                # input_size=768,
                # output_size=self.recurrent_size,
                # dropout=self.dropout,
                hidden_size=recurrent_size,
                hidden_layers=3
            )

        self.column_encoder = self._build_modules(column_encoder)
        self.table_encoder = self._build_modules(table_encoder)
        self.use_reverse = use_reverse
        if use_reverse:
            self.reversal_encoder = ReversalEncoder(recurrent_size, recurrent_size, recurrent_size, \
                n_speakers=11, gradient_clipping_bounds=10, scale_factor=0.1)
        update_modules = {
            'relational_transformer':
                spider_enc_modules.RelationalTransformerUpdate,
            'none':
                spider_enc_modules.NoOpUpdate,
        }

        self.encs_update = registry.instantiate(
            update_modules[update_config['name']],
            update_config,
            unused_keys={"name"},
            device=self._device,
            hidden_size=recurrent_size,
        )
        self.batch_encs_update = batch_encs_update


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )


    def _build_modules(self, module_types):
        module_builder = {
            'uemb':lambda: spider_enc_modules.UnitEmbeddings(
                self._device,
                self.unit_vocab,
                self.word_emb_size,
                self.learnable_words),
            'emb': lambda: spider_enc_modules.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                self.learnable_words),
            'linear': lambda: spider_enc_modules.EmbLinear(
                input_size=self.word_emb_size,
                output_size=self.word_emb_size),
            'bilstm': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False),
            'bilstm-native': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=True),
            'bilstm-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True),
            'bilstm-native-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True,
                use_native=True)
        }

        modules = []
        for module_type in module_types:
            modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward_unbatched(self, desc):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?

        # q_enc: question len x batch (=1) x recurrent_size
        # np.array(desc['audio'])
        q_enc, (_, _) = self.question_encoder([desc['question']])

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: sum of column lens x batch (=1) x recurrent_size
        c_enc, c_boundaries = self.column_encoder(desc['columns'])
        column_pointer_maps = {
            i: list(range(left, right))
            for i, (left, right) in enumerate(zip(c_boundaries, c_boundaries[1:]))
        }

        # Encode the tables
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each table into one?
        # t_enc: sum of table lens x batch (=1) x recurrent_size
        t_enc, t_boundaries = self.table_encoder(desc['tables'])
        c_enc_length = c_enc.shape[0]
        table_pointer_maps = {
            i: [
                   idx
                   for col in desc['table_to_columns'][str(i)]
                   for idx in column_pointer_maps[col]
               ] + list(range(left + c_enc_length, right + c_enc_length))
            for i, (left, right) in enumerate(zip(t_boundaries, t_boundaries[1:]))
        }

        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new now have shape
        # batch (=1) x length x recurrent_size
        q_enc_new, c_enc_new, t_enc_new = self.encs_update(
            desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)

        memory = []
        words_for_copying = []
        if 'question' in self.include_in_memory:
            memory.append(q_enc_new)
            if 'question_for_copying' in desc:
                assert q_enc_new.shape[1] == desc['question_for_copying']
                words_for_copying += desc['question_for_copying']
            else:
                words_for_copying += [''] * q_enc_new.shape[1]
        if 'column' in self.include_in_memory:
            memory.append(c_enc_new)
            words_for_copying += [''] * c_enc_new.shape[1]
        if 'table' in self.include_in_memory:
            memory.append(t_enc_new)
            words_for_copying += [''] * t_enc_new.shape[1]
        memory = torch.cat(memory, dim=1)

        return SpiderEncoderState(
            state=None,
            memory=memory,
            words=words_for_copying,
            pointer_memories={
                'column': c_enc_new,
                'table': torch.cat((c_enc_new, t_enc_new), dim=1),
            },
            pointer_maps={
                'column': column_pointer_maps,
                'table': table_pointer_maps,
            }
        )

    def forward(self, descs):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # def func(p):
        #     return p.shape[0]
        # q_enc: PackedSequencePlus, [batch, question len, recurrent_size]
        if self.use_unit:
            qs = [[desc['question_for_copying']] for desc in descs]
            q_enc, _ = self.question_encoder(qs)
        else:
            qs = [desc['question'] for desc in descs]
            max_audio_length = 0
            wav = []
            input_lens = torch.zeros((len(qs))).cuda()
            for i, q in enumerate(qs):
                input_lens[i] = q.shape[0]
                max_audio_length = max(max_audio_length,q.shape[0])
            for q in qs:
                wav_pad = np.zeros((max_audio_length-q.shape[0],q.shape[1]))
                wav.append(np.vstack([q, wav_pad]))
            wav = torch.from_numpy(np.stack(wav)).float().cuda()
            q_enc, _ = self.question_encoder(wav.transpose(0, 1), input_lens)
        # descs = sorted(descs, key=lambda sample: sample['audio'].size(1), reverse=True)
        c_enc, c_boundaries = self.column_encoder([desc['columns'] for desc in descs])
        # qs = [np.array(desc['question']) for desc in descs]
        # q_enc, (_, _) = self.question_encoder(qs)
        # longest_sample = max(qs, key=func)
        # freq_size = longest_sample.shape[1]
        # minibatch_size = len(qs)
        # max_seqlength = longest_sample.shape[0]
        # input_percentages = torch.FloatTensor(minibatch_size)
        # inputs = torch.zeros(minibatch_size, max_seqlength, freq_size).cuda()
        # for x in range(minibatch_size):
        #     tensor = torch.from_numpy(qs[x])
        #     seq_length = tensor.shape[0]
        #     inputs[x].narrow(0, 0, seq_length).copy_(tensor)
        #     input_percentages[x] = seq_length / float(max_seqlength)
        

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: PackedSequencePlus, [batch, sum of column lens, recurrent_size]

        column_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(c_boundaries_for_item, c_boundaries_for_item[1:]))
            }
            for batch_idx, c_boundaries_for_item in enumerate(c_boundaries)
        ]

        # Encode the tables
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each table into one?
        # t_enc: PackedSequencePlus, [batch, sum of table lens, recurrent_size]
        t_enc, t_boundaries = self.table_encoder([desc['tables'] for desc in descs])
 
        # c_enc_lengths = list(c_enc.orig_lengths())
        # table_pointer_maps = [
        #     {
        #         i: [
        #             idx
        #             for col in desc['table_to_columns'][str(i)]
        #             for idx in column_pointer_maps[batch_idx][col]
        #         ] +  list(range(left + c_enc_lengths[batch_idx], right + c_enc_lengths[batch_idx]))
        #         for i, (left, right) in enumerate(zip(t_boundaries_for_item, t_boundaries_for_item[1:]))
        #     }
        #     for batch_idx, (desc, t_boundaries_for_item) in enumerate(zip(descs, t_boundaries))
        # ]

        # directly point to the the table
        table_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(t_boundaries_for_item, t_boundaries_for_item[1:]))
            }
            for batch_idx, (desc, t_boundaries_for_item) in enumerate(zip(descs, t_boundaries))
        ]

        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new are PackedSequencePlus with shape
        # batch (=1) x length x recurrent_size
        spk_ids = []
        if self.use_reverse:
            logits, _ = self.reversal_encoder(q_enc, input_lens.to(self._device))
        if self.batch_encs_update:
            q_enc_new, c_enc_new, t_enc_new = self.encs_update(
                descs, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)

        result = []
        loss_sc = []
        for batch_idx, desc in enumerate(descs):
            if self.use_reverse:
                spk_ids.append(descs[batch_idx]['spk_id'])
            if self.batch_encs_update:
                q_enc_new_item = q_enc_new.select(batch_idx).unsqueeze(0)
                c_enc_new_item = c_enc_new.select(batch_idx).unsqueeze(0)
                t_enc_new_item = t_enc_new.select(batch_idx).unsqueeze(0)
            else:
                q_enc_new_item, c_enc_new_item, t_enc_new_item, align_mat_item, loss_sc_item = \
                    self.encs_update.forward_unbatched(
                        desc,
                        q_enc.select(batch_idx).unsqueeze(1) if self.use_unit else q_enc[:int(input_lens[batch_idx]),batch_idx].unsqueeze(1),
                        c_enc.select(batch_idx).unsqueeze(1),
                        c_boundaries[batch_idx],
                        t_enc.select(batch_idx).unsqueeze(1),
                        t_boundaries[batch_idx])
            # if self.training:
            #     loss_sc.append(loss_sc_item)
            memory = []
            words_for_copying = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
                # if 'question_for_copying' in desc:
                #     assert q_enc_new_item.shape[1] == len(desc['question_for_copying'])
                #     words_for_copying += desc['question_for_copying']
                # else:
                words_for_copying += [''] * q_enc_new_item.shape[1]
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new_item)
                words_for_copying += [''] * c_enc_new_item.shape[1]
            if 'table' in self.include_in_memory:
                memory.append(t_enc_new_item)
                words_for_copying += [''] * t_enc_new_item.shape[1]
            memory = torch.cat(memory, dim=1)

            result.append(SpiderEncoderState(
                state=None,
                memory=memory,
                question_memory=q_enc_new_item,
                schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                # TODO: words should match memory
                words=words_for_copying,
                pointer_memories={
                    'column': c_enc_new_item,
                    'table': torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                },
                pointer_maps={
                    'column': column_pointer_maps[batch_idx],
                    'table': table_pointer_maps[batch_idx],
                },
                m2c_align_mat=align_mat_item[0],
                m2t_align_mat=align_mat_item[1],
            ))
        if self.use_reverse and self.training:
            spk_ids = torch.from_numpy(np.array(spk_ids)).to(self._device)
            cel_loss = (torch.nn.CrossEntropyLoss()(logits, spk_ids)) * self.reversal_lambda
            return result, cel_loss
        # if self.training:
        #     return result, sum(loss_sc) / len(loss_sc)
        return result, 0.0



class Bertokens:
    def __init__(self, pieces):
        self.pieces = pieces

        self.normalized_pieces = None
        self.recovered_pieces = None
        self.idx_map = None

        self.normalize_toks()

    def normalize_toks(self):
        """
        If the token is not a word piece, then find its lemma
        If it is, combine pieces into a word, and then find its lemma
        E.g., a ##b ##c will be normalized as "abc", "", ""
        NOTE: this is only used for schema linking
        """
        self.startidx2pieces = dict()
        self.pieces2startidx = dict()
        cache_start = None
        for i, piece in enumerate(self.pieces + [""]):
            if piece.startswith("##"):
                if cache_start is None:
                    cache_start = i - 1

                self.pieces2startidx[i] = cache_start
                self.pieces2startidx[i - 1] = cache_start
            else:
                if cache_start is not None:
                    self.startidx2pieces[cache_start] = i
                cache_start = None
        assert cache_start is None

        # combine pieces, "abc", "", ""
        combined_word = {}
        for start, end in self.startidx2pieces.items():
            assert end - start + 1 < 10
            pieces = [self.pieces[start]] + [self.pieces[_id].strip("##") for _id in range(start + 1, end)]
            word = "".join(pieces)
            combined_word[start] = word

        # remove "", only keep "abc"
        idx_map = {}
        new_toks = []
        for i, piece in enumerate(self.pieces):
            if i in combined_word:
                idx_map[len(new_toks)] = i
                new_toks.append(combined_word[i])
            elif i in self.pieces2startidx:
                # remove it
                pass
            else:
                idx_map[len(new_toks)] = i
                new_toks.append(piece)
        self.idx_map = idx_map

        # lemmatize "abc"
        normalized_toks = []
        for i, tok in enumerate(new_toks):
            ann = corenlp.annotate(tok, annotators=['tokenize', 'ssplit', 'lemma'])
            lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            lemma_word = " ".join(lemmas)
            normalized_toks.append(lemma_word)

        self.normalized_pieces = normalized_toks
        self.recovered_pieces = new_toks

    def bert_schema_linking(self, columns, tables):
        question_tokens = self.normalized_pieces
        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens)

        new_sc_link = {}
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = sc_link[m_type][ij_str]

            new_sc_link[m_type] = _match
        return new_sc_link

    def bert_cv_linking(self, schema):
        question_tokens = self.recovered_pieces  # Not using normalized tokens here because values usually match exactly
        cv_link = compute_cell_value_linking(question_tokens, schema)

        new_cv_link = {}
        for m_type in cv_link:
            _match = {}
            for ij_str in cv_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = cv_link[m_type][ij_str]
            new_cv_link[m_type] = _match
        return new_cv_link


class SpiderEncoderBertPreproc(SpiderEncoderV2Preproc):

    def __init__(
            self,
            save_path,
            db_path,
            fix_issue_16_primary_keys=False,
            include_table_name_in_column=False,
            bert_version="bert-base-uncased",
            compute_sc_link=True,
            compute_cv_link=False):

        self.data_dir = os.path.join(save_path, 'enc')
        self.db_path = db_path
        self.texts = collections.defaultdict(list)
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.include_table_name_in_column = include_table_name_in_column
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link

        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

        # TODO: should get types from the data
        column_types = ["text", "number", "time", "boolean", "others"]
        self.tokenizer.add_tokens([f"<type: {t}>" for t in column_types])

    def _tokenize(self, presplit, unsplit):
        if self.tokenizer:
            toks = self.tokenizer.tokenize(unsplit)
            return toks
        return presplit

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

    def preprocess_item(self, item, validation_info):
        question = self._tokenize(item.text, item.orig['question'])
        preproc_schema = self._preprocess_schema(item.schema)
        question_bert_tokens = Bertokens(question)
        if self.compute_sc_link:
            sc_link = question_bert_tokens.bert_schema_linking(
                preproc_schema.normalized_column_names,
                preproc_schema.normalized_table_names
            )
        else:
            sc_link = {"q_col_match": {}, "q_tab_match": {}}

        if self.compute_cv_link:
            cv_link = question_bert_tokens.bert_cv_linking(item.schema)
        else:
            cv_link = {"num_date_match": {}, "cell_match": {}}

        return {
            'raw_question': item.orig['question'],
            'question': question,
            'db_id': item.schema.db_id,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
        }

    def validate_item(self, item, section):
        question = self._tokenize(item.text, item.orig['question'])
        preproc_schema = self._preprocess_schema(item.schema)

        num_words = len(question) + 2 + \
                    sum(len(c) + 1 for c in preproc_schema.column_names) + \
                    sum(len(t) + 1 for t in preproc_schema.table_names)
        if num_words > 512:
            return False, None  # remove long sequences
        else:
            return True, None

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column,
                                            self.fix_issue_16_primary_keys, bert=True)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.data_dir)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.data_dir)


    # def dataset(self, section):
    #     data = []
    #     # audio = np.load(os.path.join('/home1/liuhuadai/Transfer/huangrongjie1/projects/wav2sql/data', section+'_0_1.npy'))
    #     # # audio = np.load(f'spider/hubert/{section}_0_1.npy')
    #     # with open(os.path.join('/home1/liuhuadai/Transfer/huangrongjie1/projects/wav2sql/data', section+'_0_1.len')) as f:
    #     #     lengths = f.readlines()
    #     # sum = 0 
    #     f_r = open(os.path.join('/apdcephfs/share_1316500/nlphuang/huangrongjie1/wav2sql/processed_feature',section+'.bin'),'rb')
    #     questions = pickle.load(f_r)

    #     for i, (line,question) in enumerate(zip(open(os.path.join(self.data_dir, section + '.jsonl')),questions)):
    #         # length = int(lengths[i].strip())
    #         input = json.loads(line)
    #         # input['tables'].append(['none'])
    #         # if len(input['columns']) > 100:
    #         #     sum+=length
    #         #     continue
    #         input['speech'] = question[:700]
    #         # sum+=length
    #         data.append(input)
    #     # else:
    #     #     for i, line in enumerate(open(os.path.join(self.data_dir, section + '_skip.jsonl'))):
    #     #         input = json.loads(line)
    #     #         # if len(input['columns']) > 100:
    #     #         #     continue
    #     #         data.append(input)
    #     # # print('In total, process %d samples , skip %d extremely large databases.' % (len(data), i - len(data)))
    #     return data
    def dataset(self, section):
        data = []
        
        for i, line in enumerate(open(os.path.join(self.data_dir, section + '.jsonl'))):
            input = json.loads(line)
            # if len(input['columns']) > 100:
            #     continue
            data.append(input)
        # # print('In total, process %d samples , skip %d extremely large databases.' % (len(data), i - len(data)))
        return data

@registry.register('encoder', 'spider-bert')
class SpiderEncoderBert(torch.nn.Module):
    Preproc = SpiderEncoderBertPreproc
    batched = True

    def __init__(
            self,
            device,
            preproc,
            update_config={},
            bert_token_type=False,
            bert_version="bert-base-uncased",
            summarize_header="first",
            use_column_type=True,
            include_in_memory=('question', 'column', 'table')):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.bert_token_type = bert_token_type
        self.base_enc_hidden_size = 1024 if bert_version == "bert-large-uncased-whole-word-masking" else 768

        assert summarize_header in ["first", "avg"]
        self.summarize_header = summarize_header
        self.enc_hidden_size = self.base_enc_hidden_size
        self.use_column_type = use_column_type
        # self.lstm_modules = spider_enc_modules.LSTMs()
        self.include_in_memory = set(include_in_memory)
        update_modules = {
            'relational_transformer':
                spider_enc_modules.RelationalTransformerUpdate,
            'none':
                spider_enc_modules.NoOpUpdate,
        }

        self.encs_update = registry.instantiate(
            update_modules[update_config['name']],
            update_config,
            unused_keys={"name"},
            device=self._device,
            hidden_size=self.enc_hidden_size,
            sc_link=True,
        )
        # self.bert_model = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator")
        # self.tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
        # self.roberta_vqwav2vec = RobertaModel.from_pretrained('/home1/liuhuadai/projects/BERT-like-is-All-You-Need/checkpoints', checkpoint_file='bert_kmeans.pt')
        self.bert_model = BertModel.from_pretrained(bert_version)
        self.tokenizer = self.preproc.tokenizer
        self.bert_model.resize_token_embeddings(len(self.tokenizer))  # several tokens added

    def forward(self, descs, step=0):
        batch_token_lists = []
        batch_id_to_retrieve_question = []
        batch_id_to_retrieve_column = []
        batch_id_to_retrieve_table = []
        if self.summarize_header == "avg":
            batch_id_to_retrieve_column_2 = []
            batch_id_to_retrieve_table_2 = []
        long_seq_set = set()
        batch_id_map = {}  # some long examples are not included

        # qs = [desc['speech'] for desc in descs]
        # max_audio_length = 0
        # wav = []
        # input_lens = torch.zeros((len(qs))).cuda()
        # for i, q in enumerate(qs):
        #     input_lens[i] = q.shape[0]
        #     max_audio_length = max(max_audio_length,q.shape[0])
        # for q in qs:
        #     wav_pad = np.zeros((max_audio_length-q.shape[0],q.shape[1]))
        #     wav.append(np.vstack([q, wav_pad]))
        # wav = torch.from_numpy(np.stack(wav)).float().cuda()
        # max_qlen = max([len(desc['question_ids']) for desc in descs])
        # questions = [[0] + desc['question_ids']+[1]*(max_qlen-len(desc['question_ids'])) for desc in descs]
        # questions = torch.LongTensor(questions).to(self._device)
        # q_encs = self.roberta_vqwav2vec.extract_features(questions)
        for batch_idx, desc in enumerate(descs):
            qs = self.pad_single_sentence_for_bert(desc['question'], cls=True)
            if self.use_column_type:
                cols = [self.pad_single_sentence_for_bert(c, cls=False) for c in desc['columns']]
            else:
                cols = [self.pad_single_sentence_for_bert(c[:-1], cls=False) for c in desc['columns']]
            tabs = [self.pad_single_sentence_for_bert(t, cls=False) for t in desc['tables']]

            token_list = qs + [c for col in cols for c in col] + \
                         [t for tab in tabs for t in tab]
            # token_list = qs + [self.tokenizer.cls_token] + [c for col in cols for c in col] + \
            #              [t for tab in tabs for t in tab]
            assert self.check_bert_seq(token_list)
            if len(token_list) > 512:
                long_seq_set.add(batch_idx)
                continue

            q_b = len(qs)
            col_b = q_b + sum(len(c) for c in cols)
            # col_b = 1 + sum(len(c) for c in cols)
            # leave out [CLS] and [SEP]
            question_indexes = list(range(q_b))[1:-1]
            # use the first representation for column/table
            column_indexes = \
                np.cumsum([q_b] + [len(token_list) for token_list in cols[:-1]]).tolist()
            # column_indexes = np.cumsum([1] + [len(token_list) for token_list in cols[:-1]]).tolist()
            table_indexes = \
                np.cumsum([col_b] + [len(token_list) for token_list in tabs[:-1]]).tolist()
            if self.summarize_header == "avg":
                column_indexes_2 = \
                    np.cumsum([q_b - 2] + [len(token_list) for token_list in cols]).tolist()[1:]
                table_indexes_2 = \
                    np.cumsum([col_b - 2] + [len(token_list) for token_list in tabs]).tolist()[1:]

            indexed_token_list = self.tokenizer.convert_tokens_to_ids(token_list)
            batch_token_lists.append(indexed_token_list)

            question_rep_ids = torch.LongTensor(question_indexes).to(self._device)
            batch_id_to_retrieve_question.append(question_rep_ids)
            column_rep_ids = torch.LongTensor(column_indexes).to(self._device)
            batch_id_to_retrieve_column.append(column_rep_ids)
            table_rep_ids = torch.LongTensor(table_indexes).to(self._device)
            batch_id_to_retrieve_table.append(table_rep_ids)
            if self.summarize_header == "avg":
                assert (all(i2 >= i1 for i1, i2 in zip(column_indexes, column_indexes_2)))
                column_rep_ids_2 = torch.LongTensor(column_indexes_2).to(self._device)
                batch_id_to_retrieve_column_2.append(column_rep_ids_2)
                assert (all(i2 >= i1 for i1, i2 in zip(table_indexes, table_indexes_2)))
                table_rep_ids_2 = torch.LongTensor(table_indexes_2).to(self._device)
                batch_id_to_retrieve_table_2.append(table_rep_ids_2)

            batch_id_map[batch_idx] = len(batch_id_map)

        padded_token_lists, att_mask_lists, tok_type_lists = self.pad_sequence_for_bert_batch(batch_token_lists)
        tokens_tensor = torch.LongTensor(padded_token_lists).to(self._device)
        att_masks_tensor = torch.LongTensor(att_mask_lists).to(self._device)

        if self.bert_token_type:
            tok_type_tensor = torch.LongTensor(tok_type_lists).to(self._device)
            bert_output = self.bert_model(tokens_tensor,
                                          attention_mask=att_masks_tensor, token_type_ids=tok_type_tensor)[0]
        else:
            bert_output = self.bert_model(tokens_tensor,
                                          attention_mask=att_masks_tensor)[0]

        enc_output = bert_output

        column_pointer_maps = [
            {
                i: [i]
                for i in range(len(desc['columns']))
            }
            for desc in descs
        ]
        table_pointer_maps = [
            {
                i: [i]
                for i in range(len(desc['tables']))
            }
            for desc in descs
        ]

        assert len(long_seq_set) == 0  # remove them for now

        result = []
        for batch_idx, desc in enumerate(descs):
            c_boundary = list(range(len(desc["columns"]) + 1))
            t_boundary = list(range(len(desc["tables"]) + 1))

            if batch_idx in long_seq_set:
                q_enc, col_enc, tab_enc = self.encoder_long_seq(desc)
            else:
                bert_batch_idx = batch_id_map[batch_idx]
                # q_enc = wav[batch_idx]
                q_enc = enc_output[bert_batch_idx][batch_id_to_retrieve_question[bert_batch_idx]]
                col_enc = enc_output[bert_batch_idx][batch_id_to_retrieve_column[bert_batch_idx]]
                tab_enc = enc_output[bert_batch_idx][batch_id_to_retrieve_table[bert_batch_idx]]

                if self.summarize_header == "avg":
                    col_enc_2 = enc_output[bert_batch_idx][batch_id_to_retrieve_column_2[bert_batch_idx]]
                    tab_enc_2 = enc_output[bert_batch_idx][batch_id_to_retrieve_table_2[bert_batch_idx]]

                    col_enc = (col_enc + col_enc_2) / 2.0  # avg of first and last token
                    tab_enc = (tab_enc + tab_enc_2) / 2.0  # avg of first and last token

            # assert q_enc.size()[0] == len(desc["question_ids"])
            assert col_enc.size()[0] == c_boundary[-1]
            assert tab_enc.size()[0] == t_boundary[-1]
            # q_enc, col_enc, tab_enc = self.lstm_modules(q_enc.unsqueeze(1),
            #         col_enc.unsqueeze(1),
            #         tab_enc.unsqueeze(1),)
            q_enc_new_item, c_enc_new_item, t_enc_new_item, align_mat_item, _ = \
                self.encs_update.forward_unbatched(
                    desc,
                    q_enc.unsqueeze(1),
                    col_enc.unsqueeze(1),
                    c_boundary,
                    tab_enc.unsqueeze(1),
                    t_boundary)

            memory = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new_item)
            if 'table' in self.include_in_memory:
                memory.append(t_enc_new_item)
            memory = torch.cat(memory, dim=1)

            result.append(SpiderEncoderState(
                state=None,
                memory=memory,
                question_memory=q_enc_new_item,
                schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                # TODO: words should match memory
                words=desc['question'],
                pointer_memories={
                    'column': c_enc_new_item,
                    'table': t_enc_new_item,
                },
                pointer_maps={
                    'column': column_pointer_maps[batch_idx],
                    'table': table_pointer_maps[batch_idx],
                },
                m2c_align_mat=align_mat_item[0],
                m2t_align_mat=align_mat_item[1],
            ))
        return result, 0.0

    @DeprecationWarning
    def encoder_long_seq(self, desc):
        """
        Since bert cannot handle sequence longer than 512, each column/table is encoded individually
        The representation of a column/table is the vector of the first token [CLS]
        """
        qs = self.pad_single_sentence_for_bert(desc['question'], cls=True)
        cols = [self.pad_single_sentence_for_bert(c, cls=True) for c in desc['columns']]
        tabs = [self.pad_single_sentence_for_bert(t, cls=True) for t in desc['tables']]

        enc_q = self._bert_encode(qs)
        enc_col = self._bert_encode(cols)
        enc_tab = self._bert_encode(tabs)
        return enc_q, enc_col, enc_tab

    @DeprecationWarning
    def _bert_encode(self, toks):
        if not isinstance(toks[0], list):  # encode question words
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(toks)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][0, 1:-1]  # remove [CLS] and [SEP]
        else:
            max_len = max([len(it) for it in toks])
            tok_ids = []
            for item_toks in toks:
                item_toks = item_toks + [self.tokenizer.pad_token] * (max_len - len(item_toks))
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(item_toks)
                tok_ids.append(indexed_tokens)

            tokens_tensor = torch.tensor(tok_ids).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][:, 0, :]

    def check_bert_seq(self, toks):
        if toks[0] == self.tokenizer.cls_token and toks[-1] == self.tokenizer.sep_token:
            return True
        else:
            return False

    def pad_single_sentence_for_bert(self, toks, cls=True):
        if cls:
            return [self.tokenizer.cls_token] + toks + [self.tokenizer.sep_token]
        else:
            return toks + [self.tokenizer.sep_token]

    def pad_sequence_for_bert_batch(self, tokens_lists):
        pad_id = self.tokenizer.pad_token_id
        max_len = max([len(it) for it in tokens_lists])
        assert max_len <= 512
        toks_ids = []
        att_masks = []
        tok_type_lists = []
        for item_toks in tokens_lists:
            padded_item_toks = item_toks + [pad_id] * (max_len - len(item_toks))
            toks_ids.append(padded_item_toks)

            _att_mask = [1] * len(item_toks) + [0] * (max_len - len(item_toks))
            att_masks.append(_att_mask)

            first_sep_id = padded_item_toks.index(self.tokenizer.sep_token_id)
            assert first_sep_id > 0
            _tok_type_list = [0] * (first_sep_id + 1) + [1] * (max_len - first_sep_id - 1)
            tok_type_lists.append(_tok_type_list)
        return toks_ids, att_masks, tok_type_lists

'''
GAP version
'''

class BartTokens:
    def __init__(self, text, tokenizer):
        self.text = text
        # pieces is tokenized tokens.
        self.tokenizer = tokenizer
        self.normalized_pieces = None
        self.idx_map = None
        self.normalize_toks()

    def normalize_toks(self):
        tokens = nltk.word_tokenize(self.text.replace("'", " ' ").replace('"', ' " '))
        self.idx_map = {}
        # This map piece index to token index
        toks = []
        for i, tok in enumerate(tokens):
            self.idx_map[i] = len(toks)
            toks.extend(self.tokenizer.tokenize(tok, add_prefix_space=True))

        normalized_toks = []
        for i, tok in enumerate(tokens):
            ann = corenlp.annotate(tok, annotators=["tokenize", "ssplit", "lemma"])
            lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            lemma_word = " ".join(lemmas)
            normalized_toks.append(lemma_word)
        self.normalized_pieces = normalized_toks

    def bart_schema_linking(self, columns, tables):
        question_tokens = self.normalized_pieces
        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens)

        new_sc_link = {}
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = sc_link[m_type][ij_str]
            new_sc_link[m_type] = _match
        return new_sc_link

    def bart_cv_linking(self, schema, db_path):
        question_tokens = self.normalized_pieces
        cv_link = compute_cell_value_linking_bart(question_tokens, schema, db_path)

        new_cv_link = {}
        for m_type in cv_link:
            if m_type != "normalized_token":
                _match = {}
                for ij_str in cv_link[m_type]:
                    q_id_str, col_tab_id_str = ij_str.split(",")
                    q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                    real_q_id = self.idx_map[q_id]
                    _match[f"{real_q_id},{col_tab_id}"] = cv_link[m_type][ij_str]

                new_cv_link[m_type] = _match
            else:
                new_cv_link[m_type] = cv_link[m_type]
        return new_cv_link

def preprocess_schema_uncached_bart(schema,
                               tokenizer,
                               tokenize_func,
                               include_table_name_in_column,
                               fix_issue_16_primary_keys,
                               bart=False):
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    if bart: assert not include_table_name_in_column

    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = '<type: {}>'.format(column.type)
        if bart:
            # for bert, we take the representation of the first word
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(BartTokens(column.unsplit_name, tokenizer))
        else:
            column_name = [type_tok] + col_toks

        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1

    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bart:
            r.normalized_table_names.append(BartTokens(table.unsplit_name, tokenizer))
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r

