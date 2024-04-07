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



class facellm_test:
    def __init__(self, api_key, openai_api_base, model_name = "gpt-3.5-turbo"):
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
        SystemMessage(content=
"""
你是一个分镜师，需要根据用户给出的场景给出一个分镜的具体描述信息。
You must provide a good enough storyboard, otherwise you will be punished accordingly,
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),
        SystemMessage(content="Input: "),
        HumanMessage(content="一个男生早上起床时要做的事, 由5帧构成"), # , 请描述每一帧的场景和时间。
        SystemMessage(content="Output: "),
        AIMessage(content=
"""
{"frame": 1, "person": "1boy", "age": 24, "action": "get up, Open your eyes", "time": "morning", "scene": "bedroom"}
{"frame": 2, "person": "1boy", "age": 24, "action": "Washing and rinsing", "time": "morning", "scene": "bathroom"}
{"frame": 3, "person": "1boy", "age": 24, "action": "Eating breakfast, Sitting on a chair", "time": "morning", "scene": "kitchen"}
{"frame": 4, "person": "1boy", "age": 24, "action": "Put on formal attire", "time": "morning", "scene": "bedroom"}
{"frame": 5, "person": "1boy", "age": 24, "action": "go out, walk", "time": "morning", "scene": "street"}
"""),
        SystemMessage(content="Input: "),
        HumanMessage(content=f"{prompt}"), # , 请描述每一帧的场景和时间。
        SystemMessage(content="Output: "),
    ]
        data = self.llm(messages).content
        return [json.loads(i) for i in data.strip().split("\n")]
        
    # 通过传入故事的prompt，返回故事的多段分镜
    # input: prompt
    # output: [storyboard1, storyboard2, ...]

    def get_split_storyboard(self, prompt) -> list[str]:
        messages = [
            SystemMessage(content=
"""
你是一个分镜师，需要将用户给出的场景分成多段分镜。
You must provide a good enough storyboard, otherwise you will be punished accordingly,
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),
        SystemMessage(content="Input: "),
        HumanMessage(content="一个男生工作日一天的生活"),
        SystemMessage(content="Output: "),
        AIMessage(content='["一个男生早上起床时要做的事, 由5帧构成", "一个男生在学校上一节数学课的场景, 由5帧构成", "一个男生在家里做饭的场景, 由5帧构成"]'),
        SystemMessage(content="Input: "),
        HumanMessage(content=prompt),
        SystemMessage(content="Output: "),

        ]
        data = self.llm(messages).content
        return eval(data)

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

