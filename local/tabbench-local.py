import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from orchestra_agents import *
import agentscope
from eval_wikitq.sys_prompt import *
from local.config import get_model_configs, MODEL_SHORT_NAME, RECOMMENDED_LINE_LIMIT

# TableBench uses the WikiTQ prompt template and few-shot demos.
# The base_path is set to dataset/TableBench/ (not dataset/TabBench/ as in the
# original script, which referenced a non-existent directory).
# Make sure dataset/TableBench/prompt_template/ and dataset/TableBench/few-shot-demo/
# exist -- copy them from dataset/WikiTableQuestions/ if missing:
#   cp -r dataset/WikiTableQuestions/prompt_template dataset/TableBench/
#   cp -r dataset/WikiTableQuestions/few-shot-demo  dataset/TableBench/

dataset = pd.read_csv('dataset/TableBench/tabbench_test.tsv', sep='\t')

program = 'sql-py'
template = 'two-agent-template'
line_limit = RECOMMENDED_LINE_LIMIT

model_configs = get_model_configs(temperature=0.7)

agentscope.init(
    model_configs=model_configs,
    project="TableBench 3 agents - local vLLM",
    logger_level="ERROR",
)

with open('dataset/WikiTableQuestions/few-shot-demo/reasoning-agent-prompt.txt') as file:
    reasoner_pt = file.read()

with open('dataset/WikiTableQuestions/few-shot-demo/coding-agent-prompt.txt') as file:
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
                base_path='dataset/TableBench/',
                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                sep="\t",
                line_limit=line_limit,
                sys_pt_bank=system_pt_bank,
                few_shot_pt_bank=few_shot_pt_bank
            )
            tqa_solver.get_gpt_prediction_majority_vote(repeat_times=5)
            log_3agents, log_2agents = tqa_solver.get_log_dict()
            break
        except Exception as e:
            log_3agents = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            log_2agents = {
                'id': str(dataset.iloc[i]['id']),
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log_3agents, log_2agents
            if "DataInspectionFailed" in str(e):
                return log_3agents, log_2agents
            max_retry -= 1
    return log_3agents, log_2agents


n_threads = 1
maxLimit = float('inf')

output_result_file_3 = f'dataset/TableBench/results/3_agents/model_{MODEL_SHORT_NAME}_limit{maxLimit}.json'
output_result_file_2 = f'dataset/TableBench/results/2_agents/model_{MODEL_SHORT_NAME}_limit{maxLimit}.json'

log_3agent_list = []
log_2agent_list = []

from eval_tabbench import *
metric_eval_engine = QAMetric()

for i in tqdm(range(min(maxLimit, dataset.shape[0]))):
    log_3agents, log_2agents = parallel_func(i)
    if "uncaught_err" in log_3agents:
        print(f"uncaught error qid-{i+1}")
        continue
    log_3agent_list.append(log_3agents)
    log_2agent_list.append(log_2agents)

    if (i+1) % 10 == 0:
        tmp_result_file3 = f'dataset/TableBench/results/3_agents/tmp/model_{MODEL_SHORT_NAME}_{i+1}.json'
        json.dump(log_3agent_list, open(tmp_result_file3, 'w'), indent=4)

        ref_2agent = []
        pred_2agent = []
        for log_2a in log_2agent_list:
            if 'predicted_value' in log_2a:
                ref_2agent.append(log_2a['target_value'])
                pred_2agent.append(log_2a['predicted_value'])
        metric_scores_2 = metric_eval_engine.compute(references=ref_2agent, predictions=pred_2agent)

        ref_3agent = []
        pred_3agent = []
        for log_3a in log_3agent_list:
            if 'predicted_value' in log_3a:
                ref_3agent.append(log_3a['target_value'])
                pred_3agent.append(log_3a['predicted_value'])
        metric_scores_3 = metric_eval_engine.compute(references=ref_3agent, predictions=pred_3agent)

        print("------=")
        print("Time: ", datetime.datetime.now())
        print(f"Output file {tmp_result_file3}")
        print(f"Accuracy 2 Agent: {metric_scores_2}")
        print(f"Accuracy 3 Agent: {metric_scores_3}")
        print("------")

json.dump(log_3agent_list, open(output_result_file_3, 'w'), indent=4)

# evaluate:
ref_2agent = []
pred_2agent = []
for log_2a in log_2agent_list:
    if 'predicted_value' in log_2a:
        ref_2agent.append(log_2a['target_value'])
        pred_2agent.append(log_2a['predicted_value'])
metric_scores_2 = metric_eval_engine.compute(references=ref_2agent, predictions=pred_2agent)

ref_3agent = []
pred_3agent = []
for log_3a in log_3agent_list:
    if 'predicted_value' in log_3a:
        ref_3agent.append(log_3a['target_value'])
        pred_3agent.append(log_3a['predicted_value'])
metric_scores_3 = metric_eval_engine.compute(references=ref_3agent, predictions=pred_3agent)

print("======")
print("Time: ", datetime.datetime.now())
print(f"Output file {output_result_file_3}")
print(f"Accuracy 2 Agent: {metric_scores_2}")
print(f"Accuracy 3 Agent: {metric_scores_3}")
print("======")
