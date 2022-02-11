import math
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizerFast, AutoModelForPreTraining, AutoModelForMaskedLM

from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.dropout import WordDropout, LockedDropout
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.crf import CRFLoss
from stanza.models.common.vocab import PAD_ID
logger = logging.getLogger('stanza')
BERT_EMBEDS = 768
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NERTagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        """
        #self.model = AutoModel.from_pretrained("vinai/phobert-base").to(torch.device("cuda:0"))         
        #self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
        #self.model = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo-ner-dane")
        #self.tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo-ner-dane")
        #self.model = AutoModel.from_pretrained("flax-community/roberta-base-danish")
        #self.tokenizer = AutoModelForMaskedLM.from_pretrained("flax-community/roberta-base-danish")
        """
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        def add_unsaved_module(name, module):    
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            # load pretrained embeddings if specified
            if emb_matrix is not None:
                self.init_emb(emb_matrix)
            if not self.args.get('emb_finetune', True):
                self.word_emb.weight.detach_()
            #input_size += self.args['word_emb_dim']
            input_size += BERT_EMBEDS

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args['charlm']:
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(args['charlm_forward_file'], finetune=False))
                add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(args['charlm_backward_file'], finetune=False))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
                input_size += self.args['char_hidden_dim'] * 2

        # optionally add a input transformation layer
        if self.args.get('input_transform', False):
            self.input_transform = nn.Linear(input_size, input_size)
        else:
            self.input_transform = None
       
        # recurrent layers
        self.taggerlstm = PackedLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, \
                bidirectional=True, dropout=0 if self.args['num_layers'] == 1 else self.args['dropout'])
        # self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.drop_replacement = None
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)

        # tag classifier
        num_tag = len(self.vocab['tag'])
        self.tag_clf = nn.Linear(self.args['hidden_dim']*2, num_tag)
        self.tag_clf.bias.data.zero_()

        # criterion
        self.crit = CRFLoss(num_tag)

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])
        self.lockeddrop = LockedDropout(args['locked_dropout'])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab['word'])
        dim = self.args['word_emb_dim']
        assert emb_matrix.size() == (vocab_size, dim), \
            "Input embedding matrix must match size: {} x {}, found {}".format(vocab_size, dim, emb_matrix.size())
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        #extract bert embedding for word        
        #processed = self.extract_phobert_embeddings(self.tokenizer, self.model, word, device)
        processed = self.extract_bert_embeddings(self.tokenizer, self.model, word, device)
        words = get_float_tensor(processed, len(processed))
        assert(words[0].size(0)==tags[0].size(0))
        word = words.cuda()
        word_mask = torch.sum(torch.abs(word),2) == 0.0
        word_mask.cuda()
        if self.args['word_emb_dim'] > 0:
            word_emb = pack(torch.tensor(word))
            inputs += [word_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        if self.args['word_dropout'] > 0:
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = pad(lstm_inputs)
        lstm_inputs = self.lockeddrop(lstm_inputs)
        lstm_inputs = pack(lstm_inputs).data

        if self.input_transform:
            lstm_inputs = self.input_transform(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(\
                self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), \
                self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data


        # prediction layer
        lstm_outputs = self.drop(lstm_outputs)
        lstm_outputs = pad(lstm_outputs)
        lstm_outputs = self.lockeddrop(lstm_outputs)
        lstm_outputs = pack(lstm_outputs).data
        logits = pad(self.tag_clf(lstm_outputs)).contiguous()
        loss, trans = self.crit(logits, word_mask, tags)
        
        return loss, logits, trans
    
    def extract_bert_embeddings(self, tokenizer, model, data, device):
        """
        Extract transformer embeddings using a generic roberta extraction
        data: list of list of string (the text tokens)
        """
        #add add_prefix_space = True for RoBerTa-- error if not
        tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
        list_offsets = [[None] * (len(sentence)+2) for sentence in data]
        for idx in range(len(data)):
            offsets = tokenized.word_ids(batch_index=idx)
            for pos, offset in enumerate(offsets):
                if offset is None:
                    continue
                # this uses the last token piece for any offset by overwriting the previous value
                list_offsets[idx][offset+1] = pos
            list_offsets[idx][0] = 0
            list_offsets[idx][-1] = -1

            if len(offsets) > tokenizer.model_max_length:
                logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), data[idx])
                raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(data[idx]))

        features = []
        for i in range(int(math.ceil(len(data)/128))):
            with torch.no_grad():
                feature = model(torch.tensor(tokenized['input_ids'][128*i:128*i+128]).to(device), output_hidden_states=True)
                feature = feature[2]
                feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
                features += feature.clone().detach().cpu()

        processed = []
        #remove the bos and eos tokens
        list_offsets = [ sent[1:-1] for sent in list_offsets]
        #process the output
        for feature, offsets in zip(features, list_offsets):
            new_sent = feature[offsets]
            processed.append(new_sent)

        return processed

    def extract_phobert_embeddings(self, tokenizer, model, data, device):
        """
        Extract transformer embeddings using a method specifically for phobert
        Since phobert doesn't have the is_split_into_words / tokenized.word_ids(batch_index=0)
        capability, we instead look for @@ to denote a continued token.
        data: list of list of string (the text tokens)
        """
        processed = [] # final product, returns the list of list of word representation
        tokenized_sents = [] # list of sentences, each is a torch tensor with start and end token
        list_tokenized = [] # list of tokenized sentences from phobert
        for idx, sent in enumerate(data):
            #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
            tokenized = [word.replace("\xa0","_") for word in sent]

            #concatenate to a sentence
            sentence = ' '.join(tokenized)

            #tokenize using AutoTokenizer PhoBERT
            tokenized = tokenizer.tokenize(sentence)

            #add tokenized to list_tokenzied for later checking
            list_tokenized.append(tokenized)

            #convert tokens to ids
            sent_ids = tokenizer.convert_tokens_to_ids(tokenized)

            #add start and end tokens to sent_ids
            tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]

            if len(tokenized_sent) > tokenizer.model_max_length:
                logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(tokenized_sent), data[idx])
                #raise TextTooLongError(len(tokenized_sent), tokenizer.model_max_length, idx, " ".join(data[idx]))

            #add to tokenized_sents
            tokenized_sents.append(torch.tensor(tokenized_sent).detach())

            processed_sent = []
            processed.append(processed_sent)

            # done loading bert emb

        size = len(tokenized_sents)

        #padding the inputs
        tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents,batch_first=True,padding_value=tokenizer.pad_token_id)

        features = []

        # Feed into PhoBERT 128 at a time in a batch fashion. In testing, the loop was
        # run only 1 time as the batch size seems to be 30
        for i in range(int(math.ceil(size/128))):
            with torch.no_grad():
                feature = model(tokenized_sents_padded[128*i:128*i+128].clone().detach().to(device), output_hidden_states=True)
                # averaging the last four layers worked well for non-VI languages
                feature = feature[2]
                feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
                features += feature.clone().detach().cpu()

        assert len(features)==size
        assert len(features)==len(processed)

        #process the output
        #only take the vector of the last word piece of a word/ you can do other methods such as first word piece or averaging.
        # idx2+1 compensates for the start token at the start of a sentence
        # [0] and [-1] grab the start and end representations as well
        offsets = [[idx2+1 for idx2, _ in enumerate(list_tokenized[idx]) if (idx2 > 0 and not list_tokenized[idx][idx2-1].endswith("@@")) or (idx2==0)] 
                   for idx, sent in enumerate(processed)]
        processed = [feature[offset] for feature, offset in zip(features, offsets)]

        # This is a list of ltensors
        # Each tensor holds the representation of a sentence extracted from phobert
        return processed
