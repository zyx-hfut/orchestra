import time

import pandas as pd
import openai
import os
import json
from tqdm import tqdm
import dotenv

from dateutil import parser
from tabqa.GptPrompter import *
from tabqa.GptCOTPrompter import *
from collections import Counter

from agentscope.agents import UserAgent, DialogAgent, QAAgent


def get_token_num(s):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(s, add_special_tokens=False)
    num_tokens = len(tokens)
    return num_tokens

class CodexAnswerCOTExecutor_HighTemperaturMajorityVote(CodexAnswerCOTExecutor_template):
    def __init__(self,
                 prompt_template_json,
                 qid,
                 utterance,
                 source_csv,
                 target_value,
                 meta_data="",
                 base_path='./',
                 demo_file=None,
                 sep=',',
                 # model_config=None,
                 line_limit=float('inf'),
                 sys_pt="",
                 few_shot_pt=None):
        super().__init__(prompt_template_json, qid, utterance, source_csv,
                         target_value, base_path=base_path, demo_file=demo_file,
                         sep=sep, line_limit=line_limit)

        # TODO: model_config \in input params?
        # define the LLM config
        # self.model_config = model_config
        self.sys_pt = sys_pt
        self.few_shot_pt = few_shot_pt
        self.meta_data = meta_data
        if len(self.meta_data) > 0:
            self.meta_data = "\nNotes:" + self.meta_data + "\n"

    def _get_gpt_prediction_majority_vote(self, NNDemo=False, ft=None, repeat_times=5, maintain_df_ids=False):
        all_predictions = []
        for _ in range(repeat_times):
            """
            initialize an agent in each iteration
            """
            # pt3 = """You are a helpful assistant for tackling table-based question answering tasks."""
            agent = QAAgent(
                "TQA-"+str(self.qid),
                sys_prompt=self.sys_pt,
                model_config_name="tqa_qwen_config"
            )
            self._read_data()
            if self.few_shot_pt is None:
                self._gen_gpt_prompt(NNDemo, ft, maintain_df_ids=maintain_df_ids)
            else:
                self.get_fewshot_prompt()
            self._get_gpt_prediction(agent, maintain_df_ids=maintain_df_ids)

            all_predictions.append(self.predicted_result.strip().strip("."))
        self.all_predictions = all_predictions
        from collections import Counter
        counter = Counter(all_predictions)
        majority = counter.most_common(1)[0][0]
        self.predicted_result = majority
        # print("Answer: ", self.predicted_result)

    def get_fewshot_prompt(self):
        # todo: hard coding? pass entire few-shot prompt directly
        # self.source_table_df is assigned by invoking _read_data()
        # see GptPrompter.py --> QuestionHandler Class...
        # The value is based on source_csv, which is the actual input question

        # setup prompts for the exact input table question
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.prompt = self.prompt_template.format(data_table, self.utterance)

        # few-shot prompt + question to answer
        self.prompt = self.few_shot_pt + '\n\n' + self.prompt + self.meta_data
        # todo: test current prompt
        # print(" ||||| TEST PT |||||")
        # print(self.prompt)
        # input()

    def _log_dict(self):

        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt,
            # 'action_prompts': self.action_prompts,
            # 'decision_prompts': self.decision_prompts,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'gpt_original_output': self.original_output, 
            'all_predictions': self.all_predictions,
            'training_demo_ids': self.training_demo_ids
        }


