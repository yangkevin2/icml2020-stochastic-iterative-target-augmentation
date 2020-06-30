from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
    def __init__(self,
                 args: Namespace,
                 vocab_size: int,
                 pad_index: int = 0,
                 start_index: int = 1,
                 end_index: int = 2):
        super(Model, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.start_index = start_index
        self.end_index = end_index
        self.unconditional = args.unconditional
        self.no_vae = (not args.unconditional_vae) if args.unconditional else args.no_conditional_vae

        if self.unconditional:
            self.agg = nn.Linear(args.hidden_dim*2 + args.vae_latent_dim, args.hidden_dim)
        else:
            self.agg = nn.Linear(args.hidden_dim*4 + args.vae_latent_dim*2, args.hidden_dim)
        self.embedding = nn.Embedding(vocab_size, args.embedding_dim, padding_idx=pad_index)
        decoder_dim = args.hidden_dim * 2 + (0 if args.unconditional and args.unconditional_vae else args.vae_latent_dim)
        self.decoder_rnn = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=decoder_dim,
            num_layers=args.depth,
            bidirectional=False
        )
        self.encoder_rnn = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.depth,
            bidirectional=True
        )
        self.vae_input = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.vae_mu = nn.Linear(args.hidden_dim, args.vae_latent_dim)
        self.vae_logvar = nn.Linear(args.hidden_dim, args.vae_latent_dim)
        self.decoder_output = nn.Linear(args.hidden_dim, self.vocab_size)
        self.nonlinear = nn.ReLU()

    def _pad_mask(self, lengths: torch.LongTensor) -> torch.ByteTensor:
        # lengths: bs. Ex: [2, 3, 1]
        max_seqlen = torch.max(lengths)
        expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
        indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).cuda()  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs
    
    def _forward_lstm(self,
                      batch: torch.FloatTensor,
                      lengths: torch.LongTensor,
                      rnn: nn.LSTM,
                      init_hidden: torch.FloatTensor = None) -> torch.FloatTensor:  # init hidden should be bs x hidden if provided
        # sort lengths in descending order for pack_padded_sequence
        idx = range(lengths.size(0))
        sorted_idx = sorted(idx, key=lambda i: lengths[i], reverse=True)
        reverse_idx = [sorted_idx.index(i) for i in range(len(idx))]
        encoder_input = pack_padded_sequence(batch[:, sorted_idx, :], lengths[sorted_idx])

        if init_hidden is not None:
            init_hidden = torch.stack([init_hidden for _ in range(self.args.depth * 2)], dim=0)  # 2*depth x bs x hidden; just use same init in each direction/depth
            init_hidden = init_hidden[:, sorted_idx, :]
            enc_output, _ = rnn(encoder_input, (init_hidden, torch.zeros_like(init_hidden)))
        else:
            enc_output, _ = rnn(encoder_input)

        enc_output, _ = pad_packed_sequence(enc_output)
        enc_output = enc_output[:, reverse_idx, :]  # seqlen x bs x 2*hidden

        return enc_output
 
    def _forward_encoder(self,
                         batch_src: torch.FloatTensor,
                         lengths_src: torch.FloatTensor,
                         vae_out: torch.FloatTensor,
                         init_hidden: torch.FloatTensor = None) -> torch.FloatTensor:
        enc_output = self._forward_lstm(batch_src, lengths_src, self.encoder_rnn, init_hidden=init_hidden)
        # we don't mask out the padding here, since this is only used for attention later; we do masking at the attention step

        noisy_enc_output = torch.cat([vae_out.unsqueeze(0).repeat((enc_output.size(0), 1, 1)), enc_output], dim=2)  # seqlen x bs x 2*hidden+vae_out

        return noisy_enc_output
   
    def forward(self,
                batch_src: torch.LongTensor,
                lengths_src: torch.LongTensor,
                batch_tgt: torch.LongTensor,
                lengths_tgt: torch.LongTensor,
                props_src: torch.FloatTensor = None,
                props_tgt: torch.FloatTensor = None,
                use_vae: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        assert (props_src is None) == (props_tgt is None)

        batch_size = lengths_src.size(0)
        batch_src, batch_tgt = self.embedding(batch_src), self.embedding(batch_tgt)  # seqlen x bs x emb, dec_seqlen x bs x emb

        if self.no_vae:
            use_vae = False

        if self.unconditional:
            dec_output = self._forward_lstm(batch_tgt, lengths_tgt, self.decoder_rnn) # dec_seqlen x bs x 2*hidden
            mu, logvar = None, None
            if not self.no_vae:
                if use_vae:
                    encoded_tgt = self._forward_lstm(batch_tgt, lengths_tgt, self.encoder_rnn)
                    tgt_sums = torch.sum(encoded_tgt, dim=0)  # bs x emb, padding embeddings are 0
                    tgt_average_pools = tgt_sums / lengths_tgt.float().unsqueeze(1)  # bs x emb
                    vae_input_intermediate = self.nonlinear(self.vae_input(tgt_average_pools))
                    mu = self.vae_mu(vae_input_intermediate)
                    logvar = self.vae_logvar(vae_input_intermediate)

                    sigma = logvar.mul(0.5).exp_()
                    unit_gaussian_sample = torch.Tensor(sigma.size()).normal_().cuda()
                    sample = unit_gaussian_sample.mul(sigma).add_(mu)

                    vae_out = sample
                else:
                    vae_out = torch.zeros(batch_size, self.args.vae_latent_dim).normal_().cuda()
                dec_output = torch.cat([dec_output, vae_out.unsqueeze(0).repeat(dec_output.size(0), 1, 1)], dim=2)
            dec_output = self.nonlinear(self.agg(dec_output))
            scores = self.decoder_output(dec_output)
            return scores, mu, logvar

        if use_vae:
            encoded_src = self._forward_lstm(batch_src, lengths_src, self.encoder_rnn)
            src_sums = torch.sum(encoded_src, dim=0)  # bs x emb, padding embeddings are 0
            src_average_pools = src_sums / lengths_src.float().unsqueeze(1)  # bs x emb

            encoded_tgt = self._forward_lstm(batch_tgt, lengths_tgt, self.encoder_rnn)
            tgt_sums = torch.sum(encoded_tgt, dim=0)  # bs x emb, padding embeddings are 0
            tgt_average_pools = tgt_sums / lengths_tgt.float().unsqueeze(1)  # bs x emb

            tgt_src_diff = tgt_average_pools - src_average_pools

            vae_input_intermediate = self.nonlinear(self.vae_input(tgt_src_diff))
            mu = self.vae_mu(vae_input_intermediate)
            logvar = self.vae_logvar(vae_input_intermediate)

            sigma = logvar.mul(0.5).exp_()
            unit_gaussian_sample = torch.Tensor(sigma.size()).normal_().cuda()
            sample = unit_gaussian_sample.mul(sigma).add_(mu)

            vae_out = sample
        else:
            mu, logvar = None, None
            vae_out = torch.zeros(batch_size, self.args.vae_latent_dim).normal_().cuda()

        noisy_enc_output = self._forward_encoder(batch_src, lengths_src, vae_out)
        
        dec_output = self._forward_lstm(batch_tgt, lengths_tgt, self.decoder_rnn)  # dec_seqlen x bs x 2*hidden+vae_out

        expanded_noisy_enc_output = noisy_enc_output.unsqueeze(1).repeat((1, dec_output.size(0), 1, 1))  # seqlen x dec_seqlen x bs x 2*hidden+vae_out
        expanded_dec_output = dec_output.unsqueeze(0).repeat((noisy_enc_output.size(0), 1, 1, 1))  # seqlen x dec_seqlen x bs x 2*hidden+vae_out

        attention_weights_unnormalized = torch.sum(expanded_noisy_enc_output * expanded_dec_output, dim=3)  # seqlen x dec_seqlen x bs
        # make padding weights -inf
        padding_mask = self._pad_mask(lengths_src).float()  # seqlen x bs, 0 at pad locations
        padding_mask = padding_mask.unsqueeze(1).repeat((1, attention_weights_unnormalized.size(1), 1))  # seqlen x dec_seqlen x bs
        attention_weights_unnormalized = attention_weights_unnormalized * padding_mask - (1 - padding_mask) * 1e6  # effectively inf

        attention_weights = F.softmax(attention_weights_unnormalized, dim=0)
        attention_contexts = torch.sum(attention_weights.unsqueeze(3) * expanded_noisy_enc_output, dim=0)  # dec_seqlen x bs x 2*hidden+vae_out

        contextualized_dec_output = torch.cat([dec_output, attention_contexts], dim=-1)
        contextualized_dec_output = self.nonlinear(self.agg(contextualized_dec_output))

        scores = self.decoder_output(contextualized_dec_output)

        return scores, mu, logvar  # scores: dec_seqlen x bs x vocab, bs x vae_latent, bs x vae_latent
    
    def predict(self,
                batch_src: torch.LongTensor,
                lengths_src: torch.LongTensor,
                props_src: torch.FloatTensor = None,
                props_tgt: torch.FloatTensor = None,
                max_length: int = 100,
                sample: bool = False) -> torch.LongTensor:
        assert (props_src is None) == (props_tgt is None)

        batch_size = lengths_src.size(0)

        if not self.unconditional:
            batch_src = self.embedding(batch_src)
            vae_out = torch.zeros(batch_size, self.args.vae_latent_dim).normal_().cuda()

            noisy_enc_output = self._forward_encoder(batch_src, lengths_src, vae_out)
        if self.unconditional and not self.no_vae:
            vae_out = torch.zeros(batch_size, self.args.vae_latent_dim).normal_().cuda()

        start_seq = torch.LongTensor([self.start_index for _ in range(batch_size)]).cuda()  # bs
        end_found = torch.ByteTensor([0 for _ in range(batch_size)]).cuda()
        decoded_seq = [start_seq]
        state = None

        for _ in range(max_length):
            last_indices_embedded = self.embedding(decoded_seq[-1]).unsqueeze(0)  # 1 x bs x emb

            if state is None:  # on first iteration
                dec_output, state = self.decoder_rnn(last_indices_embedded)  # 1 x bs x 2*hidden+vae_out
            else:
                dec_output, state = self.decoder_rnn(last_indices_embedded, state)

            if self.unconditional:
                if not self.no_vae:
                    dec_output = torch.cat([dec_output, vae_out.unsqueeze(0).repeat(dec_output.size(0), 1, 1)], dim=2)
                dec_output = self.nonlinear(self.agg(dec_output.squeeze(0)))
                next_probs = F.softmax(self.decoder_output(dec_output), dim=1)
            else:
                attention_weights_unnormalized = torch.sum(noisy_enc_output * dec_output, dim=2)  # seqlen x bs
                # make padding weights -inf
                padding_mask = self._pad_mask(lengths_src).float()  # seqlen x bs, 0 at pad locations
                attention_weights_unnormalized = attention_weights_unnormalized * padding_mask - (1 - padding_mask) * 1e6  # effectively inf
                attention_weights = F.softmax(attention_weights_unnormalized, dim=0)
                attention_contexts = torch.sum(attention_weights.unsqueeze(2) * noisy_enc_output, dim=0)  # bs x 2*hidden+vae_out
                contextualized_dec_output = torch.cat([dec_output.squeeze(0), attention_contexts], dim=-1)
                contextualized_dec_output = self.nonlinear(self.agg(contextualized_dec_output))
                next_probs = F.softmax(self.decoder_output(contextualized_dec_output), dim=1)  # bs x vocab

            if sample:
                next_indices = torch.multinomial(next_probs, 1).view(-1)
            else:
                _, next_indices = next_probs.max(dim=1)  # bs

            decoded_seq.append(next_indices)

            new_end_tokens = (next_indices == self.end_index)
            end_found = end_found | new_end_tokens
            if torch.sum(end_found) == batch_size:
                break

        return decoded_seq  # list of lists of shape bs

