import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import json

# todo: 请在这里补充facellm的代码
class facellm:
    def __init__(self, api_key, model_name, openai_api_base):
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base=openai_api_base,
            openai_api_key=api_key,
            streaming=False,
        )

    # 通过传入分割后故事的prompt，返回分镜
    # input: prompt
    # output: [frame1, frame2, ...]
    # TODO: 在这里补充frame的json格式
    def get_short_storyboard(self, prompt) -> list:
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=""),
        ]
        data = self.llm(messages).content
        return data.split("\n")
        
    # 通过传入故事的prompt，返回故事的多段分镜
    # input: prompt
    # output: [storyboard1, storyboard2, ...]

    def get_split_storyboard(self, prompt) -> list[str]:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=""),
        ]
        data = self.llm(messages).content
        return data.split("\n")

    # 最外层的接口，通过传入故事的prompt，返回故事的storyboard
    # input: prompt
    # output: [frame1, frame2, ...]
    
    def get_storyboard_from_prompt(self, prompt) -> list:
        # Get the storyboard from the prompt
        split_storyboard = self.get_split_storyboard(prompt)
        storyboard = []
        for short_storyboard in split_storyboard:
            storyboard.extend(self.get_short_storyboard(short_storyboard))
        return storyboard

