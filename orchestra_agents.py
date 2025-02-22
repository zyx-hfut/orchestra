from tabqa.GptCOTPrompter import *
from agentscope.agents import QAAgent


class ThreeAgent():
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
                 line_limit=float('inf'),
                 sys_pt_bank="",
                 few_shot_pt_bank=None,
                 max_iters=5):
        self.prompt_template_json = prompt_template_json
        self.qid = qid
        self.utterance = utterance
        self.source_csv = source_csv
        self.target_value = target_value
        self.meta_data = meta_data
        self.base_path = base_path
        self.demo_file = demo_file
        self.sep = sep
        self.line_limit = line_limit
        self.sys_pt_bank = sys_pt_bank # dict: {"agent type": sys_prompt}
        self.intermediate_table_df = None
        self.few_shot_pt_bank = few_shot_pt_bank
        self.max_iters = max_iters

    def _init_agents(self):
        self.reasoning_agent = ReasoningAgent(
            self.prompt_template_json,
            self.qid,
            self.utterance,
            self.source_csv,
            self.target_value,
            self.meta_data,
            self.base_path,
            self.demo_file,
            self.sep,
            self.line_limit,
            self.sys_pt_bank["reasoning"],
            self.few_shot_pt_bank["reasoner_pt"]
        )
        self.coding_agent = CodingAgent(
            self.prompt_template_json,
            self.qid,
            self.utterance,
            self.source_csv,
            self.target_value,
            self.base_path,
            self.demo_file,
            self.sep,
            self.line_limit,
            self.sys_pt_bank["coding"],
            self.few_shot_pt_bank["coder_pt"]
        )

    def get_gpt_prediction_majority_vote(self, NNDemo=False, ft=None, repeat_times=5, maintain_df_ids=False):
        all_predictions_2agents = []
        all_predictions = []
        for _ in range(repeat_times):
            try:
                """
                initialize pair of reasoning and coding agents in each iteration
                """
                # setup all agents
                self._init_agents()

                # reasoner-coder interaction
                self._get_prediction(max_iterations=self.max_iters)
                all_predictions_2agents.append(self.predicted_result_2agents)
                # invoke decision agent
                self.predicted_result = self._get_final_answer()
                all_predictions.append(self.predicted_result)
            #input(f"2agents: {all_predictions_2agents}\n3agents: {all_predictions}\n")
            except Exception as e:
                pass

        # majority vote (i.e., self-consistency)
        self.all_predictions = all_predictions
        self.predicted_result = self.get_majority(self.all_predictions)

        self.all_predictions_2agents = all_predictions_2agents
        self.predicted_result_2agents = self.get_majority(self.all_predictions_2agents)

        # print("Target Value:", self.target_value)
        # print(" ======== Reasoning Agent ======== ")
        # self.reasoning_agent.agent.get_msg_tqa()
        # print("\n ======== Coding Agent ======== ")
        # self.coding_agent.agent.get_msg_tqa()
        # print("##################\n##################\n")
        # input()

    def _get_prediction(self, max_iterations=5):
        for _ in range(max_iterations):
            reasoning_msg = self.reasoning_agent._get_response()
            if reasoning_msg['type'] == "Answer":
                self.predicted_result_2agents = reasoning_msg['content']
                return
            else: # type can be either Answer or Instruction
                # coding agent receives the reasonging agent's instructions
                # update current prompt
                self.coding_agent._receive_instruction(reasoning_msg['content'])

                # coding agent generate and execute codes; return the intermediate table
                intermediate_table_df = self.coding_agent._get_intermediate_tb()

                # reasoning agent get the updated intermediate table
                # update current prompt
                self.reasoning_agent._receive_table(intermediate_table_df)

        # reach the max iteraions, enfore to get an answer
        self.reasoning_agent.get_answer = True
        reasoning_msg = self.reasoning_agent._get_response()
        self.predicted_result_2agents = reasoning_msg['content']
        # self.predicted_result = reasoning_msg['content']

    def _get_final_answer(self):
        # get reasoning path
        reasoning_msg_list = self.reasoning_agent.get_reasoning_path()

        # setup decision agent
        self.decision_agent = DecisionAgent(
                                sys_pt=self.sys_pt_bank["decision_agent"],
                                reasoning_path_msg=reasoning_msg_list)
        decision_agent_answer = self.decision_agent.get_final_answer()

        # self.decision_agent.agent.get_msg()
        return decision_agent_answer

    def get_log_dict(self):
        log_3agents = {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'all_predictions': self.all_predictions,
            # 'prompt': self.prompt,
            ## 'action_prompts': self.action_prompts,
            ## 'decision_prompts': self.decision_prompts,
            # 'execution_match': self.execution_acc,
            # 'gpt_error': self.gpt_error,
            # 'execution_err': self.execution_err,
            # 'predicted_sql': self.predicted_sql,
            # 'df_reformat_sql': self.reformat_sql,
            # 'training_demo_ids': self.training_demo_ids
        }
        log_2agents = {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result_2agents,
            'all_predictions': self.all_predictions_2agents,
        }
        return log_3agents, log_2agents

    def get_majority(self, prediction_list):
        from collections import Counter
        counter = Counter(prediction_list)
        majority = counter.most_common(1)[0][0]
        return majority

    def get_all_logs(self):
        print("   *****  REASONING AGENT   *****  ")
        self.reasoning_agent.agent.get_msg_tqa()
        print("   *****  CODING AGENT   *****  ")
        self.coding_agent.agent.get_msg_tqa()
        print("============\n============\n\n\n")


class ReasoningAgent(CodexAnswerCOTExecutor):
    def __init__(
        self,
        prompt_template_json,
        qid,
        utterance,
        source_csv,
        target_value,
        meta_data="", # empty by default
        base_path='./',
        demo_file=None,
        sep=',',
        line_limit=float('inf'),
        sys_pt=None,
        few_shot_pt=None
    ):
        super().__init__(qid, utterance, source_csv, target_value, base_path, demo_file, sep)
        self.prompt_template_dict = json.load(open(os.path.join(self.base_path, prompt_template_json)))
        self.prompt_template = self.prompt_template_dict['prompt_template_reasoning']
        self.line_limit = line_limit # 10
        self.original_output = []
        self.current_prompt = "" # initial prompt should be table+question
        self.get_answer = False # enfore to generate answer
        self.demo_prompt = few_shot_pt
        self.meta_data = meta_data
        self.agent = QAAgent(
            name="reasoner",
            sys_prompt=sys_pt,
            model_config_name="reasoning_agent"
        )
        self._read_data()
        self._gen_gpt_prompt()
        if len(self.meta_data) > 0:
            self.meta_data = "\nNotes:" + self.meta_data + "\n"
        ####
        # self.agent.get_msg()
        # input()
        # self._get_gpt_prediction(agent, maintain_df_ids=maintain_df_ids)

    def _gen_gpt_prompt(self):
        # observe few-shot prompt
        # the first msg in agent.memory would be few-shot examples
        few_shot_msg = prompt2messages(self.demo_prompt)
        self.agent.observe(few_shot_msg)

        # setup prompts for the exact input table question
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.prompt = self.prompt_template.format(data_table, self.utterance)

        # input the question
        self.current_prompt = self.prompt + self.meta_data

    def _get_response(self):
        """
        Given current prompt (initial table questions or intermediate tables)
        Return LLM output: either an Instruction or the Answer
        todo: add output into log var
        """
        # when iteration reaches its maximum, set get_answer=True
        if self.get_answer:
            self.current_prompt = "Please provide an answer directly based on current information, starting with 'Answer: '"
            original_output = GptCompletion(self.agent, self.current_prompt)
            original_result = original_output['text'].strip('\n')
            answer_type = "Answer"
            answer_content = self._get_answer_content(original_result)
            response = {'type': answer_type, 'content': answer_content}
            return response

        original_output = GptCompletion(self.agent, self.current_prompt)
        # self.agent.get_msg()
        # remove leading/ending "\n"s
        original_result = original_output['text'].strip('\n')
        self.original_output.append(original_result)

        if "Instruction:" in original_result:
            answer_type = "Instruction"
            answer_content = self._get_instruction_content(original_result)  # Get the content after "Instruction: "
        elif "Answer:" in original_result:
            answer_type = "Answer"
            answer_content = self._get_answer_content(original_result)  # Get the content after "Answer: "
        else: # invalid type, enfore to get an answer
            self.current_prompt = "Please provide an answer directly based on current information, starting with 'Answer: '"
            original_output = GptCompletion(self.agent, self.current_prompt)
            original_result = original_output['text'].strip('\n')
            answer_type = "Answer"
            answer_content = self._get_answer_content(original_result)
        response = {'type': answer_type, 'content': answer_content}
        return response

    def _get_answer_content(self, s):
        lines = s.split('\n')
        # Iterate through each line to find the one that starts with "Answer:"
        ans_content = ""
        for line in lines:
            if "Answer:" in line:
                # Extract the content after "Answer:"
                ans_content = line.split("Answer:")[-1].strip().strip(".")
                break
        return ans_content

    def _get_instruction_content(self, s):
        # lines = s.split('\n')
        # # Iterate through each line to find the one that starts with "Answer:"
        # ins_content = ""
        # for line in lines:
        #     if "Instruction:" in line:
        #         # Extract the content after "Answer:"
        #         ins_content = line.split("Instruction:")[-1].strip()
        #         break
        ins_content = s.split("Instruction:")[-1].strip()
        return ins_content

    def _receive_table(self, df):
        # setup current prompts, upon received the intermediate table...
        # coding agent produce table in dataframe (df) format
        if df is None:
            self.get_answer = True
            return
        data_table = table_formater(df, permute_df=False, line_limit=self.line_limit)
        inter_tb_template = self.prompt_template_dict['intermediate_prompt_template']['Reasoner']
        self.current_prompt = inter_tb_template.format(data_table)

    def get_reasoning_path(self):
        # get reasoning history (remove few-shot prompt and final answer)
        return self.agent.get_reasoning_path()


class CodingAgent(CodexAnswerCOTExecutor):
    def __init__(
        self,
        prompt_template_json,
        qid,
        utterance,
        source_csv,
        target_value,
        base_path='./',
        demo_file=None,
        sep=',',
        line_limit=float('inf'),
        sys_pt=None,
        few_shot_pt=None
    ):
        super().__init__(qid, utterance, source_csv, target_value, base_path, demo_file, sep)

        self.prompt_template_dict = json.load(open(os.path.join(self.base_path, prompt_template_json)))
        self.prompt_template = self.prompt_template_dict['prompt_template_coding']
        self.line_limit = line_limit # 10
        self.code_history = []
        self.original_output = []
        self.current_prompt = "" # initial prompt should be table+question+instruction
        self.intermediate_table_df = None
        self.demo_prompt = few_shot_pt
        # initialize coding agent here
        self.agent = QAAgent(
            name="coder",
            sys_prompt=sys_pt,
            model_config_name="coding_agent"
        )
        self._read_data()
        self._gen_gpt_prompt()
        # print("TEST CODING PT:\n", self.current_prompt)

    def _gen_gpt_prompt(self):
        # self.source_table_df is assigned by invoking _read_data()
        # see GptPrompter.py --> QuestionHandler Class...
        # The value is based on source_csv, which is the actual input question
        # few-shot prompt
        # self.prompt = self.demo_prompt + '\n\n'
        # self.current_prompt = self.prompt
        few_shot_msg = prompt2messages(self.demo_prompt+'\n\n')
        self.agent.observe(few_shot_msg)

    def _receive_instruction(self, instruction):
        """ update the prompt based on the received coding instructions """
        # for first iteration; prompt_template = {original table} + {instruction}
        if self.intermediate_table_df is None:
            self.intermediate_table_df = self.source_table_df
            data_table = table_formater(self.intermediate_table_df, permute_df=False, line_limit=self.line_limit)
            self.current_prompt = self.prompt_template.format(data_table, instruction)
        # intermediate iterations; prompt template = {intermediate table} + {instruction}
        else:
            data_table = table_formater(self.intermediate_table_df, permute_df=False, line_limit=self.line_limit)
            inter_tb_template = self.prompt_template_dict['intermediate_prompt_template']['Coder']
            self.current_prompt = inter_tb_template.format(data_table, instruction)

    def _get_intermediate_tb(self):
        original_output = GptCompletion(self.agent, self.current_prompt)
        # remove leading/ending "\n"s
        original_result = original_output['text'].strip('\n')
        code_type = original_result.split(":")[0]
        code_content = original_result.split('```')[1]

        renewed_df = self._executor(self.intermediate_table_df, code_content, code_type)

        i = len(self.series_dfs) - 1
        while i >= 0 and (renewed_df is None):  # or renewed_df.shape[0] == 0):
            self.intermediate_table_df = self.series_dfs[i]
            renewed_df = self._executor(self.intermediate_table_df, code_content, code_type)
            if renewed_df is not None:
                self.gpt_error = None
            i -= 1
        self.intermediate_table_df = renewed_df

        # If generated table is invalid or generate repeated results
        # break the loop and enfore the reasoning agent to output the answer
        if self.intermediate_table_df is None or code_content in self.code_history:
            return None

        self.code_history.append(code_content)
        self.series_dfs.append(renewed_df)
        return self.intermediate_table_df


class DecisionAgent():
    """
    Take reasoning history from reasoning agent, generate the final answer.
    """
    def __init__(self, sys_pt=None, reasoning_path_msg=None):
        self.agent = QAAgent(
            name="DecisionAgent",
            sys_prompt=sys_pt,
            model_config_name="reasoning_agent"
        )
        # setup memory for decision agent
        self.agent.load_memory(reasoning_path_msg)
        # check memory; for debug
        # self.agent.get_msg()

    def get_final_answer(self):
        # prompt to get the final answer
        self.current_prompt = "Please provide an answer directly based on current information, starting with 'Answer: '"
        original_output = GptCompletion(self.agent, self.current_prompt)
        original_result = original_output['text'].strip().strip(".")
        # return original_result.split(':')[1].strip().strip(".")

        s = original_output['text'].strip('\n')
        lines = s.split('\n')
        # Iterate through each line to find the one that starts with "Answer:"
        ans_content = ""
        get_ans = False
        for line in lines:
            if "Answer:" in line:
                # Extract the content after "Answer:"
                ans_content = line.split("Answer:")[-1].strip().strip(".")
                get_ans = True
                break
        if not get_ans:
            ans_content = original_result.split(':')[1].strip().strip(".")
        return ans_content
