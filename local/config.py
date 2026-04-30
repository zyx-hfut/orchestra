import os

# === vLLM Server Configuration ===
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:9876/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3-4B-Instruct")

# === Model Display Name (for result file naming) ===
MODEL_SHORT_NAME = "qwen3-4b-instruct"

# === Context Window Consideration ===
# Qwen3-4B-Instruct has ~32K context window (vs Qwen2.5-14B's ~128K).
# The original line_limit=float('inf') may cause tables to exceed context.
RECOMMENDED_LINE_LIMIT = 100


def get_model_configs(temperature=0.7):
    """Build AgentScope model configs for vLLM OpenAI-compatible API.

    config_name values ("reasoning_agent", "coding_agent") are hardcoded
    in orchestra_agents.py and must not be changed.
    """
    cfg = {
        "model_type": "openai_chat",
        "model_name": VLLM_MODEL_NAME,
        "api_key": VLLM_API_KEY,
        "client_args": {
            "base_url": VLLM_BASE_URL,
        },
        "generate_args": {
            "temperature": temperature,
        },
    }
    return [
        {**cfg, "config_name": "reasoning_agent"},
        {**cfg, "config_name": "coding_agent"},
    ]
