from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

class InternLM2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
        return response, history



class OpenAILikeChat(BaseModel):
    def __init__(self,api_key:str,base_url:str,model_name) -> None:
        super().__init__()
        self.client = OpenAI(api_key=api_key,base_url=base_url)
        self.model_name = model_name

    def chat(self,prompt:str,history:List[dict], system_prompt:str ='') ->Tuple[str,List[dict]]:
        message = []

        if system_prompt:
            message.append({"role": "system", "content": system_prompt})

        message.extend(history)

        message.append({"role": "user","content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0.1
        )

        reply_text = response.choices[0].message.content

        new_history = history + [
            {"role": "user","content": prompt},
            {"role": "assistant","content":reply_text}
        ]

        return reply_text,new_history


# ========================================================
# 2. 智谱 GLM API 接入层
# ========================================================
class ZhipuChat(BaseModel):
    def __init__(self, api_key: str, model: str = "glm-4") -> None:
        super().__init__('')
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def chat(self, prompt: str, history: List[dict], system_prompt: str = '') -> Tuple[str, List[dict]]:
        # GLM 的处理逻辑与 OpenAI 类似
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        reply_text = response.choices[0].message.content

        new_history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply_text}
        ]
        return reply_text, new_history




# if __name__ == '__main__':
#     model = InternLM2Chat('/root/share/model_repos/internlm2-chat-7b')
#     print(model.chat('Hello', []))