from transformers import BartModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
# import pl_bolts
import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from allennlp.training.metrics import CategoricalAccuracy
from src.utils import chunks

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
        self.transformer = T5ForConditionalGeneration.from_pretrained(t5_type)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_type)
        self._accuracy = CategoricalAccuracy()
        
    def metric_reset(self):
        self._accuracy.reset()
        
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
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    # warm up lr
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        # update params
        optimizer.step(closure=closure)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    

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
                        self._accuracy(sample['false/true'].unsqueeze(0), label)
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
                print(input_text)
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
    def __init__(self, args, t5_type='t5-small'):
        super().__init__()
        self.lr = getattr(args, "lr")
        self.epochs = getattr(args, "epochs")
        self.warmup_steps = getattr(args, "warmup_steps")
        self.transformer = T5_Cond_Gen_Wrapper.from_pretrained(t5_type)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_type)
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(t5_type)
        self.decoder_tokenizer.padding_side = 'left' # necessary since initial decoding sequences could have different length
                 
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.lm_head = self.transformer.lm_head
        self._accuracy = CategoricalAccuracy()
    
    def sample_to_target_text(self, sample):
        # we use lower since it is one token for true and false
        if 'decoder_text' in sample:
            return sample['decoder_text']
        elif 'answer' not in sample:
            return f"<pad>Claim: {str(sample['question'])} Answer:"
        else:
            return f"<pad>Claim: {str(sample['question'])} Answer: {str(sample['answer']).lower()}</s>"
    
    def samples_to_input(self, input_samples):
        fusion_map = []
        flat_sample_text = []
        for s, i in zip(input_samples, range(len(input_samples))):
            fusion_map.append([len(flat_sample_text), len(flat_sample_text)+len(s['facts'])])
            flat_sample_text += s['facts']
        return flat_sample_text, fusion_map
    
    def encoder_forward(self, fusion_map, input_ids, attention_mask):
        embed_dim = self.transformer.config.hidden_size
        batch_size = len(fusion_map)
        encoder_outputs = self.transformer.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs[0]
        
        longest_fused_seq = max([attention_mask[start:end].sum() for start, end in fusion_map])
        encoder_fused_states = torch.zeros((batch_size, longest_fused_seq, embed_dim), device=self.device)
        fused_attention_mask = torch.zeros((batch_size, longest_fused_seq), device=self.device)
        
        fused_encoder_states = []
        for (start, end), i in zip(fusion_map, range(batch_size)):
            selected_states = encoder_hidden_states[start:end]
            encoder_attention_mask = attention_mask[start:end].reshape(-1).to(torch.bool)
            flat_encoder_states = selected_states.reshape(-1,embed_dim)[encoder_attention_mask]
            
            encoder_fused_states[i,:flat_encoder_states.shape[0]] = flat_encoder_states
            fused_attention_mask[i,:flat_encoder_states.shape[0]] = 1
        
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_fused_states,
                hidden_states= None,
                attentions=fused_attention_mask
            )
        return encoder_outputs
    
    def forward(self, fusion_map, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        encoder_outputs = self.encoder_forward(fusion_map, input_ids, attention_mask)
        encoder_fused_states=encoder_outputs.last_hidden_state
        fused_attention_mask=encoder_outputs.attentions
        
        print('encoder_fused_states', encoder_fused_states.shape)
        print('fused_attention_mask', fused_attention_mask.shape)
        print('decoder_input_ids', decoder_input_ids.shape)
        print('decoder_attention_mask', decoder_attention_mask.shape)
        
        dec_outputs = self.decoder(input_ids=decoder_input_ids, 
                    attention_mask=decoder_attention_mask, 
                    encoder_hidden_states=encoder_fused_states, 
                    encoder_attention_mask=fused_attention_mask)
        sequence_output = dec_outputs[0]
        print(sequence_output)
        lm_logits = self.lm_head(sequence_output)
        
        return lm_logits
    
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
                                   use_cache=False)  
                
        loss_fct = nn.CrossEntropyLoss()
#         print(logits.reshape(-1, self.transformer.config.vocab_size), decoder_target.reshape(-1))
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
        batch_target_text = [self.sample_to_target_text(s) for s in input_samples]
        
        encoder_tok_obj = self.tokenizer(batch_input_text, return_tensors='pt', padding=True)
        collated_samples['encoder_ids'] = encoder_tok_obj['input_ids']
        collated_samples['encoder_att_mask'] = encoder_tok_obj['attention_mask']
        
        decoder_tok_obj = self.decoder_tokenizer(batch_target_text, return_tensors='pt', padding=True, add_special_tokens=False)
        collated_samples['decoder_input_ids'] = decoder_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = decoder_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = decoder_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
    def inference(self, input_samples, max_len=30, chunk_size=64, num_return_sequences=1, **kwargs):
        """
        input_samples: [{'all_raw_queries':['sadfad','adfad'], ...}]
        """
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="Re-writing"):
                flat_sample_text, fusion_map = self.samples_to_input(input_samples)
                encoder_tok_obj = self.tokenizer(flat_sample_text, return_tensors='pt', padding=True)
                input_ids = encoder_tok_obj['input_ids'].to(self.device)
                attention_mask = encoder_tok_obj['attention_mask'].to(self.device)
                
                encoder_outputs = self.encoder_forward(fusion_map, input_ids, attention_mask)
                fused_attention_mask=encoder_outputs.attentions
                
                batch_target_text = [self.sample_to_target_text(s) for s in input_samples]
                decoder_tok_obj = self.decoder_tokenizer(batch_target_text, return_tensors='pt', padding=True, add_special_tokens=False)
                decoder_input_ids = decoder_tok_obj['input_ids'].to(self.device)
                decoder_attention_mask = decoder_tok_obj['attention_mask'].to(self.device)
                                
                kwargs.update({'encoder_outputs':encoder_outputs, 'decoder_attention_mask':decoder_attention_mask})
                
                outputs = self.transformer.generate(decoder_input_ids, 
                                                       attention_mask=fused_attention_mask, 
                                                       num_return_sequences=num_return_sequences,
                                                       num_beams=num_return_sequences, 
                                                       max_length=max_len, 
                                                       early_stopping=True, 
                                                       return_dict_in_generate=True,
                                                       output_scores=True,
                                                       use_cache=False,
                                                       **kwargs)
                output_ids = outputs.sequences
                output_scores = outputs.sequences_scores if num_return_sequences>1 else torch.tensor([1.0]*len(chunk_samples))
                output_text = [self.tokenizer.decode(single_output_ids, skip_special_tokens=False) for single_output_ids in output_ids]
                output_chunks = list(chunks(output_text, num_return_sequences))
                output_score_chunks = list(chunks(output_scores, num_return_sequences))

                for sample, all_gen_per_sample, scores in zip(chunk_samples, output_chunks, output_score_chunks):
                    sample['all_generations'] = all_gen_per_sample
                    sample['scores'] = scores.softmax(-1)
                    sample['top_output'] = all_gen_per_sample[0]
                new_samples += chunk_samples
            return new_samples