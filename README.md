# Orchestra
Code implementation of Orchestra, a LLM-powered multi-agent framework for table question answering (TQA).

## Files
- ```agentscope/```: A flexible multi-agent platform. This is cloned from the [AgentScope](https://github.com/modelscope/agentscope) project, with necessary modifications to adapt it to the TQA task.
- ```tabqa/```: Contains the base class for LLM agents. This is cloned from the [ReAcTable](https://github.com/yunjiazhang/ReAcTable) repo, with necessary modifications to adapt it to the multi-agent-based TQA framework.
- ```dataset/```: Contains TQA benchmarks along with the corresponding few-shot prompts for each benchmark.
- ```orchestra_agents.py```: Contains the code implementation of the orchestra agents.

## Installation
Note: This repo requires Python 3.9 and is built upon the [AgentScope](https://github.com/modelscope/agentscope) and [ReAcTable](https://github.com/yunjiazhang/ReAcTable) projects.
We have modified the source code of both AgentScope and ReAcTable. Therefore, it is recommended to install these projects from the source code in this repository instead of setting them up directly from their original GitHub repositories.

1. Setup conda environment:

```bash
conda create -n orchestra python=3.9
```

2. Install AgentScope from source:
```bash
# Install the package in editable mode
cd agentscope
pip install -e .
```

3. Install tabqa from source:
```bash
cd tabqa
pip install -e .
```

4. Additional note on required dependencies:

Ensure that the following dependencies are installed to prevent conflicts between AgentScope and ReAcTable, and to enable agents to execute SQL queries correctly:
```
Flask-SQLAlchemy==3.0.2
sqlalchemy==1.4.46
pandas==1.5.3
pandasql==0.7.3
```

## Usage
We provide usage examples in ```wikitq-qwen14b.py``` and ```tabfact-qwen14b.py```. 

**Step 1**: Specify the Backbone LLM for the Agent

```python
llm_model = "qwen2.5-14b-instruct"
```

**Step 2**: Set up Agent Configuration for the AgentScope Platform

```python
"""
Configurations for the logic agent (reasoning_agent) and query agent (coding_agent).
Here, we use DashScope API in our example.
Note: AgentScope supports both local model services (e.g., Ollama) and third-party model APIs (e.g., OpenAI API).
Modify the configurations as needed to use other LLMs.
"""
model_configs = [
    {
        "config_name": "reasoning_agent",
        "model_type": "dashscope_chat",
        "model_name": llm_model,
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),  # load from environment variable
    },
    {
        "config_name": "coding_agent",
        "model_type": "dashscope_chat",
        "model_name": llm_model,
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),  # load from environment variable
    }
]

agentscope.init(
    model_configs=model_configs,  # model_config,
    project="WikiTQ",
    # logger_level="ERROR",
)
```

**Step 3**: Set Up Other Components
```python
# load few-shot prompts for in-context learning
with open('dataset/WikiTableQuestions/few-shot-demo/reasoning-agent-prompt.txt') as file:
    reasoner_pt = file.read()

with open('dataset/WikiTableQuestions/few-shot-demo/coding-agent-prompt.txt') as file:
    coder_pt = file.read()

few_shot_pt_bank = {
    "reasoner_pt": reasoner_pt,
    "coder_pt": coder_pt
}

n_threads = 1  # number of threads
maxLimit = 20  # number of test cases
```

**Step 4**: Run the Orchestra Framework

The Orchestra framework will be instantiated and executed on the i-th TQA test case by invoking the following function:

```python
parallel_func(i)
```

The evaluation results will be stored in:
```bash
dataset/{benchmark-name}/results/
```

## Credits
We adopt the [AgentScope](https://github.com/modelscope/agentscope) project to implement our multi-agent TQA framework and use the [ReAcTable](https://github.com/yunjiazhang/ReAcTable) framework to instantiate the ReAct paradigm for LLM agents.
