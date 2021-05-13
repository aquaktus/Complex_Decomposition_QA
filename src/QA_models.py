from transformers import BartModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, T5TokenizerFast, T5ForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
# import pl_bolts
import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import os
from allennlp.training.metrics import CategoricalAccuracy
from src.utils import chunks

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import ipywidgets as ipyw

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput

class T5_QA_From_Oracle_Facts(LightningModule):
    def __init__(self, args, t5_type='t5-base'):
        """
        R1 = Raw 1
        
        Training:
        R1 + R2 + R3 -> M3
        """
        super().__init__()
        self.lr = getattr(args, "lr")
        self.epochs = getattr(args, "epochs")
        self.warmup_steps = getattr(args, "warmup_steps")
        self.gpu_id = getattr(args, "gpu_id")
        self.transformer = T5ForConditionalGeneration.from_pretrained(t5_type)
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_type)
        self.EM_accuracy = CategoricalAccuracy()
        self.to('cpu' if self.gpu_id==-1 else f"cuda:{self.gpu_id}")
        
    def metric_reset(self):
        self.EM_accuracy.reset()
        
    def forward(self, encoder_input, decoder_input):
        outputs = self.transformer(input_ids, decoder_input_ids=decoder_input)
        return outputs
    
    def sample_to_input_text(self, sample):
        return f"Context: {' ||| '.join(sample['facts'])} ||| Query: {sample['question']}<extra_id_0>"
    
    def sample_to_target_text(self, sample):
        # we use lower since it is one token for true and false
        return f"<extra_id_0>{str(sample['answer']).lower()}"
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        batch_input_text = [self.sample_to_input_text(s) for s in input_samples]
        batch_target_text = [self.sample_to_target_text(s) for s in input_samples]
        
        encoder_tok_obj = self.tokenizer(batch_input_text, return_tensors='pt', padding=True)
        collated_samples['encoder_ids'] = encoder_tok_obj['input_ids']
        collated_samples['encoder_att_mask'] = encoder_tok_obj['attention_mask']
        
        decoder_tok_obj = self.tokenizer(batch_target_text, return_tensors='pt', padding=True)
        collated_samples['decoder_input_ids'] = decoder_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = decoder_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = decoder_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch["encoder_ids"]
        input_mask = batch['encoder_att_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        decoder_mask = batch['decoder_att_mask']
                
        outputs = self.transformer(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
#         print(logits.reshape(-1, self.transformer.config.vocab_size), decoder_target.reshape(-1))
        loss = loss_fct(logits.reshape(-1, self.transformer.config.vocab_size), decoder_target.reshape(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return loss
    
#     def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
#     # warm up lr
#         if self.trainer.global_step < self.warmup_steps:
#             lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warmup_steps))
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr_scale * self.lr

#         # update params
#         optimizer.step(closure=closure)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    def save(self, save_dir):
        checkpoint_path = os.path.join(save_dir, 'model.sate_dict')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)
    
    def load(self, save_dir):
        checkpoint_path = os.path.join(save_dir, 'model.sate_dict')
        self.load_state_dict(torch.load(checkpoint_path))
    
    def QA_inference(self, input_samples, chunk_size=64):
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="QA-inference"):
                batch_size = len(chunk_samples)
                input_text = [self.sample_to_input_text(s) for s in chunk_samples]
                input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
                input_ids = input_tok_obj['input_ids'].to(self.device)
                input_att_mask = input_tok_obj['attention_mask'].to(self.device)
                
                decoder_start_token_id = self.tokenizer.get_vocab()['<extra_id_0>']
                decoder_ids = torch.full((batch_size,1), decoder_start_token_id).to(self.device)
                
                outputs = self.transformer(input_ids, 
                            decoder_input_ids=decoder_ids, 
                            attention_mask=input_att_mask, 
                            use_cache=False)
                logits = outputs[0]
                
                true_token_id = self.tokenizer.get_vocab()['▁true']   # careful "▁" is not an underscore "_"
                false_token_id = self.tokenizer.get_vocab()['▁false']
                
                for sample, token_logits in zip(chunk_samples, logits):
                    true_logit = token_logits[0][true_token_id]
                    false_logit = token_logits[0][false_token_id]
                    sample['false/true'] = torch.tensor([false_logit, true_logit]).softmax(-1)
                    if 'answer' in sample:
                        label = torch.tensor([sample['answer']]) # tensor [True/False]
                        self.EM_accuracy(sample['false/true'].unsqueeze(0), label)
                new_samples += chunk_samples
                
            return new_samples
    
    def inference(self, input_samples, max_len=200, chunk_size=64, num_return_sequences=4, **kwargs):
        """
        input_samples: [{'all_raw_queries':['sadfad','adfad'], ...}]
        """
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="Re-writing"):
                input_text = [self.sample_to_input_text(s) for s in chunk_samples]
                input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
                input_ids = input_tok_obj['input_ids'].to(self.device)
                input_att_mask = input_tok_obj['attention_mask'].to(self.device)
                
                decoder_start_token_id = self.tokenizer.get_vocab()['<extra_id_0>']
                
                outputs = self.transformer.generate(input_ids, 
                                                       attention_mask=input_att_mask, 
                                                       num_return_sequences=num_return_sequences,
                                                       num_beams=max(4, num_return_sequences), 
                                                       max_length=max_len, 
                                                       early_stopping=True, 
                                                       return_dict_in_generate=True,
                                                       output_scores=True,
                                                       decoder_start_token_id=decoder_start_token_id, 
                                                       **kwargs)
                output_ids = outputs.sequences
                output_scores = outputs.sequences_scores
                output_text = [self.tokenizer.decode(single_output_ids, skip_special_tokens=True) for single_output_ids in output_ids]
                output_chunks = list(chunks(output_text, num_return_sequences))
                output_score_chunks = list(chunks(output_scores, num_return_sequences))

                for sample, all_gen_per_sample, scores in zip(chunk_samples, output_chunks, output_score_chunks):
                    sample['all_generations'] = all_gen_per_sample
                    sample['scores'] = scores.softmax(-1)
                    sample['top_output'] = all_gen_per_sample[0]
                new_samples += chunk_samples
            return new_samples
        
class T5_QA_Only_Query(T5_QA_From_Oracle_Facts):
    def sample_to_input_text(self, sample):
        return f"Query: {sample['question']}<extra_id_0>"
    
class T5_Cond_Gen_Wrapper(T5ForConditionalGeneration):
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, decoder_attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        
        padding_delta = input_ids.shape[1] - decoder_attention_mask.shape[1]
        batch_size = input_ids.shape[0]
        
        new_decoder_mask = torch.cat([decoder_attention_mask, torch.ones(batch_size, padding_delta, device=self.device)], dim=1)

        return {
            "decoder_input_ids": input_ids,
            # (input_ids!=self.config.pad_token_id).to(torch.float),
            "decoder_attention_mask": new_decoder_mask,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        return input_ids
    
    
class Reasoning_in_Decoder(LightningModule):
    def __init__(self, args, t5_type='t5-base'):
        """
        R1 = Raw 1
        
        Training:
        R1 + R2 + R3 -> M3
        """
        super().__init__()
        self.lr = getattr(args, "lr")
        self.epochs = getattr(args, "epochs")
        self.warmup_steps = getattr(args, "warmup_steps")
        self.gpu_id = getattr(args, "gpu_id")
        self.transformer = T5_Cond_Gen_Wrapper.from_pretrained(t5_type)
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_type)
        self.EM_accuracy = CategoricalAccuracy()
        self.to('cpu' if self.gpu_id==-1 else f"cuda:{self.gpu_id}")
   
        self.decoder_tokenizer = T5TokenizerFast.from_pretrained(t5_type)
        self.decoder_tokenizer.padding_side = 'left' # necessary since initial decoding sequences could have different length
        
        self.validation_scores = []
                 
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.lm_head = self.transformer.lm_head
    
    def sample_to_train_target_text(self, sample):
        # we use lower since it is one token for true and false
        if 'decoder_text' in sample:
            return sample['decoder_text']
        else:
            return f"<pad> Claim: {str(sample['question'])} Proof: {sample['proof']} Answer: {sample['answer']}</s>"
    
    def sample_to_inference_target_text(self, sample):
        if 'decoder_text' in sample:
            return sample['decoder_text']
        return f"<pad> Claim: {str(sample['question'])}"
    
    def samples_to_input(self, input_samples):
        fusion_map = []
        flat_sample_text = []
        for s, i in zip(input_samples, range(len(input_samples))):
            fusion_map.append([len(flat_sample_text), len(flat_sample_text)+len(s['facts'])])
            flat_sample_text += s['facts']
        return flat_sample_text, fusion_map
    
    def metric_reset(self):
        self.EM_accuracy.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    def save(self, save_dir, save_file_name='model.sate_dict'):
        checkpoint_path = os.path.join(save_dir, save_file_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)
    
    def load(self, save_path):
        self.load_state_dict(torch.load(save_path))
    
    def encoder_forward(self, fusion_map, input_ids, attention_mask, return_hidden_states=False):
        embed_dim = self.transformer.config.hidden_size
        batch_size = len(fusion_map)
        encoder_outputs = self.transformer.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_hidden_states, return_dict=True)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        longest_fused_seq = max([attention_mask[start:end].sum() for start, end in fusion_map])
        encoder_fused_states = torch.zeros((batch_size, longest_fused_seq, embed_dim), device=self.device)
        fused_attention_mask = torch.zeros((batch_size, longest_fused_seq), device=self.device)
        
        layer_fused_encoder_states = []
        if return_hidden_states:
            encoder_layers_hidden_states = encoder_outputs.hidden_states
            layers = len(encoder_layers_hidden_states)
            encoder_layers_fused_states = torch.zeros((batch_size, longest_fused_seq, layers, embed_dim), device=self.device)
            for (start, end), i in zip(fusion_map, range(batch_size)):
                encoder_layers_hidden_states = torch.einsum('ijkl->jkil', torch.stack(encoder_layers_hidden_states)) if isinstance(encoder_layers_hidden_states, tuple) else encoder_layers_hidden_states
                selected_states = encoder_layers_hidden_states[start:end]
                
                encoder_attention_mask = attention_mask[start:end].reshape(-1).to(torch.bool)
                
                flat_encoder_layer_states = selected_states.reshape(-1,layers, embed_dim)[encoder_attention_mask]
                encoder_layers_fused_states[i,:flat_encoder_layer_states.shape[0]] = flat_encoder_layer_states
        
        fused_encoder_states = []
        for (start, end), i in zip(fusion_map, range(batch_size)):
            selected_states = encoder_hidden_states[start:end]
            encoder_attention_mask = attention_mask[start:end].reshape(-1).to(torch.bool)
            flat_encoder_states = selected_states.reshape(-1,embed_dim)[encoder_attention_mask]
            
            encoder_fused_states[i,:flat_encoder_states.shape[0]] = flat_encoder_states
            fused_attention_mask[i,:flat_encoder_states.shape[0]] = 1
        
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_fused_states,
                hidden_states= encoder_layers_fused_states if return_hidden_states else None,
                attentions=fused_attention_mask
            )
        return encoder_outputs
    
    def forward(self, fusion_map, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, return_hidden_states=False, **kwargs):
        encoder_outputs = self.encoder_forward(fusion_map, input_ids, attention_mask)
        encoder_fused_states=encoder_outputs.last_hidden_state
        fused_attention_mask=encoder_outputs.attentions
        encoder_layer_states=encoder_outputs.hidden_states
        
        dec_outputs = self.decoder(input_ids=decoder_input_ids, 
                    attention_mask=decoder_attention_mask, 
                    encoder_hidden_states=encoder_fused_states, 
                    encoder_attention_mask=fused_attention_mask,
                    output_hidden_states=return_hidden_states)
        sequence_output = dec_outputs[0]
        lm_logits = self.lm_head(sequence_output)
        
        return Seq2SeqLMOutput(logits=lm_logits, 
                               encoder_hidden_states=encoder_layer_states)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["encoder_ids"]
        attention_mask = batch['encoder_att_mask']
        fusion_map = batch['fusion_map']
        
        decoder_input_ids = batch['decoder_input_ids']
        decoder_target_ids = batch['decoder_target_ids']
        decoder_attention_mask = batch['decoder_att_mask']
                
        logits = self.forward(fusion_map=fusion_map,
                                   input_ids=input_ids, 
                                   attention_mask=attention_mask,
                                   decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_attention_mask,
                                   use_cache=False).logits  
                
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.transformer.config.pad_token_id)
        loss = loss_fct(logits.reshape(-1, self.transformer.config.vocab_size), decoder_target_ids.reshape(-1))
        if torch.isnan(loss):
            print(f'Got NaN my dude...')
            
        return loss
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        batch_input_text, fusion_map = self.samples_to_input(input_samples)
        collated_samples = {'fusion_map':fusion_map}
        batch_target_text = [self.sample_to_train_target_text(s) for s in input_samples]
        
        encoder_tok_obj = self.tokenizer(batch_input_text, return_tensors='pt', padding=True)
        collated_samples['encoder_ids'] = encoder_tok_obj['input_ids']
        collated_samples['encoder_att_mask'] = encoder_tok_obj['attention_mask']
        
        decoder_tok_obj = self.decoder_tokenizer(batch_target_text, return_tensors='pt', padding=True, add_special_tokens=False)
        collated_samples['decoder_input_ids'] = decoder_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = decoder_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = decoder_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
    def inference(self, input_samples, max_len=30, chunk_size=64, num_return_sequences=1, return_hidden_states=False, **kwargs):
        """
        input_samples: [{'all_raw_queries':['sadfad','adfad'], ...}]
        """
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="Inference"):
                flat_sample_text, fusion_map = self.samples_to_input(chunk_samples)
                encoder_tok_obj = self.tokenizer(flat_sample_text, return_tensors='pt', padding=True)
                input_ids = encoder_tok_obj['input_ids'].to(self.device)
                attention_mask = encoder_tok_obj['attention_mask'].to(self.device)
                
                encoder_outputs = self.encoder_forward(fusion_map, input_ids, attention_mask, return_hidden_states=return_hidden_states)
                fused_attention_mask=encoder_outputs.attentions
                encoder_layer_states=encoder_outputs.hidden_states if return_hidden_states else [None]*len(chunk_samples)
                
                batch_target_text = [self.sample_to_inference_target_text(s) for s in chunk_samples]
                decoder_tok_obj = self.decoder_tokenizer(batch_target_text, return_tensors='pt', padding=True, add_special_tokens=False)
                decoder_input_ids = decoder_tok_obj['input_ids'].to(self.device)
                decoder_attention_mask = decoder_tok_obj['attention_mask'].to(self.device)
                                
                kwargs.update({'encoder_outputs':encoder_outputs, 'decoder_attention_mask':decoder_attention_mask})
                
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    if input_ids.shape[0] < decoder_input_ids[batch_id].shape[0]:
                        return decoder_input_ids[batch_id][input_ids.shape[0]].tolist()
                    else:
                        return list(range(self.decoder_tokenizer.vocab_size))
                
                outputs = self.transformer.generate(decoder_input_ids, 
                                                       attention_mask=fused_attention_mask, 
                                                       num_return_sequences=num_return_sequences,
                                                       num_beams=num_return_sequences, 
                                                       max_length=max_len, 
                                                       early_stopping=True, 
                                                       output_hidden_states=True,
                                                       return_dict_in_generate=True,
                                                       output_scores=True,
                                                       prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                       use_cache=False,
                                                       **kwargs)
                output_ids = outputs.sequences
                output_scores = outputs.sequences_scores if num_return_sequences>1 else torch.tensor([1.0]*len(chunk_samples))
                output_text = [self.tokenizer.decode(single_output_ids, skip_special_tokens=True) for single_output_ids in output_ids]
                output_chunks = list(chunks(output_text, num_return_sequences))
                output_score_chunks = list(chunks(output_scores, num_return_sequences))
                
                decoder_layer_states=torch.einsum('ijkl->jkil', torch.stack(outputs.decoder_hidden_states[-1])) if return_hidden_states else [None]*len(chunk_samples)

                for i in range(len(chunk_samples)):
                    start, end = fusion_map[i]
                    chunk_samples[i]['encoder_input_ids'] = input_ids[start:end]
                    chunk_samples[i]['decoder_input_ids'] = output_ids[i]
                    chunk_samples[i]['all_generations'] = output_chunks[i]
                    chunk_samples[i]['scores'] = output_score_chunks[i].softmax(-1)
                    chunk_samples[i]['top_output'] = output_chunks[i][0]
                    chunk_samples[i]['encoder_hidden_states'] = encoder_layer_states[i]
                    chunk_samples[i]['decoder_hidden_states'] = decoder_layer_states[i]
                    if 'answer' in chunk_samples[i]:
                        target_text = self.sample_to_train_target_text(chunk_samples[i])
                        target_text = self.decoder_tokenizer.decode(self.decoder_tokenizer.encode(target_text, add_special_tokens=False), skip_special_tokens=True)
                        chunk_samples[i]['target_text'] = target_text
                        is_same = chunk_samples[i]['top_output']==target_text
                        self.EM_accuracy(torch.tensor([[not is_same, is_same]]).to(torch.float), torch.tensor([1]))
                        chunk_samples[i]['EM'] = is_same
                        
                new_samples += chunk_samples
            return new_samples
        
    def visualise(self, sample):
        
        encoder_ids = sample['encoder_input_ids'].flatten()
        decoder_ids = sample['decoder_input_ids']
        all_ids = encoder_ids.tolist() + decoder_ids.tolist()

        encoder_tokens = self.tokenizer.batch_decode(encoder_ids.view(-1,1))
        decoder_tokens = self.tokenizer.batch_decode(decoder_ids.view(-1,1))
        all_tokens = encoder_tokens + decoder_tokens
        all_token_and_ids = [f"{idx}->{tok}" for idx, tok in zip(all_ids, all_tokens)]
        
        ipyw.interact(self.visualise_for_token, sample=ipyw.fixed(sample), target_id=ipyw.SelectionSlider(
            options=all_token_and_ids,
            value=all_token_and_ids[0],
            description='Selected token:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        ))
        
    def visualise_for_token(self, sample, target_id):
        target_id = int(target_id.split('->')[0]) if isinstance(target_id, str) else target_id
        sample = self.inference([sample], return_hidden_states=True)[0]
        
        encoder_dots = self.transformer.lm_head(sample['encoder_hidden_states']).argsort(descending=True).tolist()
        decoder_dots = self.transformer.lm_head(sample['decoder_hidden_states']).argsort(descending=True).tolist()
        
        encoder_ids = sample['encoder_input_ids'].flatten()
        decoder_ids = sample['decoder_input_ids']

        encoder_tokens = self.tokenizer.batch_decode(encoder_ids.view(-1,1))
        decoder_tokens = self.tokenizer.batch_decode(decoder_ids.view(-1,1))

        layers = sample['encoder_hidden_states'].shape[1]
        encoder_plot = torch.zeros((layers, len(encoder_ids)), dtype=torch.int)
        decoder_plot = torch.zeros((layers, len(decoder_ids)-1), dtype=torch.int)

        # encoder 
        for row in tqdm(range(layers)):
            for column in range(len(encoder_ids)):
                rank = encoder_dots[column][row].index(target_id)     #(input_ids[column+1])
                encoder_plot[row, column] = rank

        # decoder
        for row in tqdm(range(layers)):
            for column in range(len(decoder_ids)-1):
                rank = decoder_dots[column][row].index(target_id)     #(input_ids[column+1])
                decoder_plot[row, column] = rank
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        h1 = ax1.imshow(encoder_plot+1, norm=LogNorm(), cmap=plt.cm.GnBu_r, aspect='auto')
        h2 = ax2.imshow(decoder_plot+1, norm=LogNorm(), cmap=plt.cm.GnBu_r, aspect='auto')

        ax1.xaxis.set_ticks(range(len(encoder_tokens)))
        ax1.set_xticklabels(encoder_tokens, minor=False, rotation = 45)

        # Loop over data dimensions and create text annotations.
        for i in range(layers):
            for j in range(len(encoder_tokens)):
                if encoder_plot[i, j] < 200:
                    text = ax1.text(j, i, int(encoder_plot[i, j]),
                                   ha="center", va="center", color="w")

        ax2.xaxis.set_ticks(range(len(decoder_tokens)))
        ax2.set_xticklabels(decoder_tokens, minor=False, rotation = 45)
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()

        ax1.set_title("Encoder")
        ax2.set_title("Decoder")
        fig.suptitle(f"T5 logit similarity rank for tok '{self.tokenizer.decode([target_id])}'", fontsize=16, y=1.1)

        fig.colorbar(h2, ax=ax2)
        plt.show(block=False) 
        
class Reasoning_in_Encoder_Qestion_in_Decoder(Reasoning_in_Decoder):
    def samples_to_input(self, input_samples):
        fusion_map = []
        flat_sample_text = []
        for s, i in zip(input_samples, range(len(input_samples))):
            fusion_map.append([len(flat_sample_text), len(flat_sample_text)+1])
            flat_sample_text += [' | '.join(s['facts'])]
        return flat_sample_text, fusion_map
    
class Fusion_in_Decoder(Reasoning_in_Decoder):
    def sample_to_train_target_text(self, sample):
        # we use lower since it is one token for true and false
        if 'decoder_text' in sample:
            return sample['decoder_text']
        else:
            return f"<pad> Proof: {sample['proof']} Answer: {sample['answer']}</s>"
    
    def sample_to_inference_target_text(self, sample):
        if 'decoder_text' in sample:
            return sample['decoder_text']
        return f"<pad>"
    
    def samples_to_input(self, input_samples):
        fusion_map = []
        flat_sample_text = []
        for s, i in zip(input_samples, range(len(input_samples))):
            fusion_map.append([len(flat_sample_text), len(flat_sample_text)+len(s['facts'])])
            question = s['question']
            flat_sample_text += [f"Claim:{question} | {fact}" for fact in s['facts']]
        return flat_sample_text, fusion_map
    
class Standard_Transformer(Reasoning_in_Decoder):
    def sample_to_train_target_text(self, sample):
        # we use lower since it is one token for true and false
        if 'decoder_text' in sample:
            return sample['decoder_text']
        else:
            return f"<pad> Answer: {sample['answer']} Proof: {sample['proof']} </s>"
    
    def sample_to_inference_target_text(self, sample):
        if 'decoder_text' in sample:
            return sample['decoder_text']
        return f"<pad>"
    
    def samples_to_input(self, input_samples):
        fusion_map = []
        flat_sample_text = []
        for s, i in zip(input_samples, range(len(input_samples))):
            question = s['question']
            fusion_map.append([len(flat_sample_text), len(flat_sample_text)+1])
            flat_sample_text += [f"Claim: {question} ||| Facts: " + ' | '.join(s['facts'])]
        return flat_sample_text, fusion_map