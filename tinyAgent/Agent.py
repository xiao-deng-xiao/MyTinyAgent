from typing import Dict, List, Optional, Tuple, Union
import json5

# ❌ 删掉这行，不需要强依赖本地模型了
# from tinyAgent.LLM import InternLM2Chat
from tinyAgent.tool import Tools

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    # ✅ 修改 1：不再传入 path，而是传入一个 model_engine（模型引擎对象）
    def __init__(self, model_engine) -> None:
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        # ✅ 修改 2：直接使用外部传进来的模型
        self.model = model_engine

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:'): j].strip()
            plugin_args = text[j + len('\nAction Input:'): k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        response, his = self.model.chat(text, history, self.system_prompt)
        # 调试用，打印模型的第一轮思考
        print(f"[{self.model.__class__.__name__} 思考中]:\n", response)

        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            print(f"\n[Agent 正在调用工具] {plugin_name} 参数: {plugin_args}")
            response += self.call_plugin(plugin_name, plugin_args)
            print(f"[工具返回结果]: {response.split('Observation:')[-1]}")

        # 注意：这里你之前写的 his 传对了！非常棒！
        response, his = self.model.chat(response, his, self.system_prompt)
        return response, his