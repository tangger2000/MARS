from pathlib import Path
import os
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor


# Define your API keys and configurations
OPENAI_API_KEY = "sk-4aJj5ygdQ9rw6lS6920712Ef9bB848439522E72318439eCd"
# OPENAI_BASE_URL = "http://154.44.26.195:17935/v1"
OPENAI_BASE_URL = "http://159.75.70.95:17935/v1"
# OPENAI_BASE_URL = "https://vip.apiyi.com/v1"

# MODEL = "chatgpt-4o-latest"
MODEL = "gpt-4o-2024-11-20"
# MODEL = "deepseek-chat"
# config_list = [{"model": MODEL, "api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL, "temperature": 0}]
config_list = [{"model": MODEL, "api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL}]

SILENT = True # 关闭嵌套智能体的输出
STREAM = True # stream on console
CACHE = None # None 就是关闭 41是默认值开启

GRAPH_RAG = "/root/data50T/LYT/matagent/backend/psk-graphrag/settings.yaml"

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".coding")
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)
# code_executor = DockerCommandLineCodeExecutor(bind_dir=Path(WORK_DIR))