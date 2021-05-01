import json
import re

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
                with open(f'data/big_files/rule-reasoning-dataset-V2020.2.5.0/original/depth-{hop_type}/{split}.jsonl', 'r') as rt_f:
                    for line in rt_f:
                        situation = json.loads(line)
                        facts = [fact.strip() for fact in re.findall('.*?[.]', situation['context']) ]
                        for question_dict in situation['questions']:
                            self.data[f'rule_taker_{hop_type}'][split].append({
                                'question':question_dict['text'],
                                'facts':facts,
                                'answer':question_dict['label']
                            })
    
    def get_data(self, dataset_name, split):
        return self.data[dataset_name][split]
    