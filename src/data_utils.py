import json
import re
import random
import string 
import itertools

flatten = itertools.chain.from_iterable

class Claim_Verification_Dataset():
    def __init__(self):
        """
        StrategyQA has a very nice format for claim verification. 
        Each sample is composed of:
        - q_id: str
        - term: str
        - description: str
        - question: str
        - answer: bool
        - facts: [str]
        - decomposition: [str]
        - evidence: [dict]
        """
        self.data = {}
        
        #get data for strategyQA
        self.data['strategyQA'] = {}
        for split in ['train', 'dev']:
            with open(f'data/small_files/StrategyQA/{split}.json', 'r') as sqa_f:
                self.data['strategyQA'][split] = json.load(sqa_f)
        
        for hop_type in ['0', '1', '2', '3', '3ext', '3ext-NatLang', '5']:
            self.data[f'rule_taker_{hop_type}'] = {}
            for split in ['train', 'dev', 'test']:
                self.data[f'rule_taker_{hop_type}'][split] = []
                with open(f'data/big_files/rule-reasoning-dataset-V2020.2.5.0/original/depth-{hop_type}/meta-{split}.jsonl', 'r') as rt_f:
                    for i, line in enumerate(rt_f):
                        situation = json.loads(line)
                        triples = [f"{triple_id}: {value['text']}" for triple_id, value in situation['triples'].items()]
                        rules = [f"{rule_id}: {value['text']}" for rule_id, value in situation['rules'].items()]
                        facts = triples + rules
                        random.shuffle(facts)
                        for q_id, question_dict in situation['questions'].items():
                            if 'CWA' in question_dict['proofs'] or 'NAF' in question_dict['proofs']:
                                proof = 'Not enough information'
                                answer = 'none'
                            else:
                                proof = re.sub('[\(\)\[\]]+', '', question_dict['proofs']).split(' OR ')[-1]
                                answer = 'true' if question_dict['answer'] else 'false'
                            self.data[f'rule_taker_{hop_type}'][split].append({
                                'q_id':f"{i}{q_id}",
                                'question':question_dict['question'],
                                'facts':facts,
                                'answer':answer,
                                'proof':proof
                            })
                    d_set = self.data[f'rule_taker_{hop_type}'][split]
                    true_with_not_in_q = [s for s in d_set if s['answer'] == 'true' and 'not' in s['question']]
                    true_with_no_not_in_q = [s for s in d_set if s['answer'] == 'true' and not 'not' in s['question']]
                    false_with_not_in_q = [s for s in d_set if s['answer'] == 'false' and 'not' in s['question']]
                    false_with_no_not_in_q = [s for s in d_set if s['answer'] == 'false' and not 'not' in s['question']]
                    none_with_not_in_q = [s for s in d_set if s['answer'] == 'none' and 'not' in s['question']]
                    none_with_no_not_in_q = [s for s in d_set if s['answer'] == 'none' and not 'not' in s['question']]
                    subsets = [true_with_not_in_q, true_with_no_not_in_q, false_with_not_in_q, false_with_no_not_in_q, none_with_not_in_q, none_with_no_not_in_q]
                    smallest_subset_len = min(map(len,subsets))
                    self.data[f'rule_taker_{hop_type}'][f"{split}_balanced"] = []
                    for subset in subsets:
                        self.data[f'rule_taker_{hop_type}'][f"{split}_balanced"] += subset[:smallest_subset_len]
                    random.shuffle(self.data[f'rule_taker_{hop_type}'][f"{split}_balanced"])
    
    def get_data(self, dataset_name, split):
        return self.data[dataset_name][split]
    
class Variable_Mapping_Dataset():
    def __init__(self, size=1000, triples_range=[2,3,4], hops_range=[0,1], distractors_range=[5]):
        self.data_map = {}
        samples = []
        self.triples_range = triples_range
        self.hops_range = hops_range
        self.distractors_range = distractors_range
        self.repetition_cache = set()
        
        for i in range(size):
            samples.append(self.new_sample())
            
        l = len(samples)
        train_split_idx = int(l*0.8)
        val_split_idx = int(l*0.9)
        self.data_map['all'] = samples
        self.data_map['train'] = samples[:train_split_idx]
        self.data_map['dev'] = samples[train_split_idx:val_split_idx]
        self.data_map['test'] = samples[val_split_idx:]  
        
    def new_sample(self):
        entity_pool = list(string.ascii_uppercase)
        value_pool = list(string.ascii_lowercase)
        random.shuffle(entity_pool)
        random.shuffle(value_pool)
        num_triples = random.choice(self.triples_range)
        triple_entities = entity_pool[:num_triples]
        del entity_pool[:num_triples]
        triple_values = value_pool[:num_triples]
        del value_pool[:num_triples]
        hop_maps = [triple_entities]
        num_hops = random.choice(self.hops_range)
        for i in range(num_hops):
            hop_maps.append(entity_pool[:num_triples])
            del entity_pool[:num_triples]
        
        facts = []
        for entity, value in zip(triple_entities, triple_values):
            facts.append(f"{entity} = '{value}'")
        for sources, targets in zip(hop_maps[:-1], hop_maps[1:]):
            for source_entity, target_entity in zip(sources, targets):
                facts.append(f"{source_entity} = {target_entity}")
        for distractor_entity in entity_pool[:random.choice(self.distractors_range)]:
            source_entity = random.choice(list(flatten(hop_maps)))
            facts.append(f"{source_entity} = {distractor_entity}")
        random.shuffle(facts)
            
        sample = {'facts':facts}
        sample['question'] = f"What is the value of {hop_maps[-1][0]}?"
        sample['answer'] = triple_values[0]
        proof = [m[0] for m in hop_maps[::-1]]
        sample['proof'] = ' = '.join(proof) + f" = {triple_values[0]}"
        
        return sample