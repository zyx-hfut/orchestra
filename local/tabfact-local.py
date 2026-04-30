import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from orchestra_agents import *
import agentscope
from eval_tabfact.sys_prompt import *
from local.config import get_model_configs, MODEL_SHORT_NAME, RECOMMENDED_LINE_LIMIT

dataset = pd.read_csv('dataset/TabFact/small_test.tsv', sep='\t')

program = 'sql-py'
template = 'two-agent-template'
line_limit = RECOMMENDED_LINE_LIMIT

model_configs = get_model_configs(temperature=0.7)

agentscope.init(
    model_configs=model_configs,
    project="TabFact 3 agents - local vLLM",
    logger_level="ERROR",
)

with open('dataset/TabFact/few-shot-demo/reasoning-agent-prompt.txt') as file:
    reasoner_pt = file.read()

with open('dataset/TabFact/few-shot-demo/coding-agent-prompt.txt') as file:
    coder_pt = file.read()

few_shot_pt_bank = {
    "reasoner_pt": reasoner_pt,
    "coder_pt": coder_pt
}

def parallel_func(i):
    max_retry = 3
    while max_retry > 0:
        try:
            tqa_solver = ThreeAgent(
                f'prompt_template/{template}.json',
                dataset.iloc[i]['id'],
                dataset.iloc[i]['utterance'],
                dataset.iloc[i]['context'],
                dataset.iloc[i]['targetValue'],
                base_path='dataset/TabFact/',
                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                sep="#",
                line_limit=line_limit,
                sys_pt_bank=system_pt_bank,
                few_shot_pt_bank=few_shot_pt_bank
            )
            tqa_solver.get_gpt_prediction_majority_vote(repeat_times=5)
            log_3agents, _ = tqa_solver.get_log_dict()
            break
        except Exception as e:
            log_3agents = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log_3agents
            max_retry -= 1
    return log_3agents


n_threads = 1
maxLimit = 20

output_result_file_3 = f'dataset/TabFact/results/3_agents/model_{MODEL_SHORT_NAME}_limit{maxLimit}.json'

log_3agent_list = []

correct_cnt = 0
total_cnt = 0
for i in tqdm(range(min(maxLimit, dataset.shape[0]))):
    log_3agents = parallel_func(i)
    log_3agent_list.append(log_3agents)
    total_cnt += 1

    if 'predicted_value' in log_3agents and log_3agents['target_value'] in log_3agents['predicted_value'].lower():
        correct_cnt += 1

    if (i+1) % 5 == 0:
        tmp_result_file3 = f'dataset/TabFact/results/3_agents/tmp/model_{MODEL_SHORT_NAME}_{i+1}.json'
        json.dump(log_3agent_list, open(tmp_result_file3, 'w'), indent=4)

        print("------=")
        print("Time: ", datetime.datetime.now())
        print(f"Output file {tmp_result_file3}")
        print(f"Examples: {total_cnt}\nCorrect: {correct_cnt}\nAccuracy: {correct_cnt/total_cnt}")
        print("------")


json.dump(log_3agent_list, open(output_result_file_3, 'w'), indent=4)

print("======")
print("Time: ", datetime.datetime.now())
print(f"Output file {output_result_file_3}")
print(f"Examples: {total_cnt}\nCorrect: {correct_cnt}\nAccuracy: {correct_cnt / total_cnt}")
print("======")
