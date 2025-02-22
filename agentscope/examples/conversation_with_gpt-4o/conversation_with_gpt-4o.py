# -*- coding: utf-8 -*-
"""An example for conversation with OpenAI vision models, especially for
GPT-4o."""
import agentscope
from agentscope.agents import UserAgent, DialogAgent

import os


# Fill in your OpenAI API key
YOUR_OPENAI_API_KEY = "xxx"

model_config__ = {
    "config_name": "gpt-4o_config",
    "model_type": "openai_chat",
    "model_name": "gpt-4o",
    "api_key": YOUR_OPENAI_API_KEY,
    "generate_args": {
        "temperature": 0.7,
    },
}

model_config_ = {
    "model_type": "post_api_chat",
    "config_name": "gpt_postapi_config",
    "api_url": f"{os.environ.get('HTTP_LLM_URL')}",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + f"{os.environ.get('HTTP_LLM_API_KEY')}"
    },
    "messages_key": "messages",
    "json_args": {
        "model": "gpt-4o"  # "gpt-4o"
    },
}


model_config = {
    "config_name": "llama",
    "model_type": "openai_chat",
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "api_key": "9XwiaxuH96GJQPbJXb5WGMomPi1foaXv",
    "client_args": {
        "base_url": "https://api.deepinfra.com/v1/openai",
    },
    "generate_args": {
        "temperature": 0.7,
    },
}


agentscope.init(
    model_configs=model_config,
    project="Conversation with GPT-4o",
)

# Require user to input URL, and press enter to skip the URL input
user = UserAgent("user", require_url=True)

agent = DialogAgent(
    "Friday",
    sys_prompt="You are a super nice assistant.",
    model_config_name="llama",
)

x = None
while True:
    x = agent(x)
    x = user(x)
    if x.content == "exit":  # type "exit" to break the loop
        break
