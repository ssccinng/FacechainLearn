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
            max_tokens=1024
        )

    # 通过传入分割后故事的prompt，返回分镜
    # input: prompt
    # output: [frame1, frame2, ...]
    # TODO: 在这里补充frame的json格式
    def get_short_storyboard(self, prompt, lastContent, theme = "") -> list:
        
        messages = [
        SystemMessage(content=
f"""
你是一个分镜师，需要根据用户给出的场景给出一个分镜的具体描述信息.
请你需要遵从逻辑，保持每个场景的连贯性，包括年龄动作等.
person只能是1boy,1girl,1women,1man等直接关于性别的描述,年轻时用1boy或者1girl,年长时使用1man或者1women，不能包含职业等信息
除了例子之外，用户给出的输入是连续的分镜，请你注意上下文的逻辑关系, 比如年龄是否增长，动作的连贯性等.
每个场景只会由一个人物出现.
年龄大于10的时候输出10的倍数，比如20,30,40,50,60等.
用户场景的主题是{theme}，请你依据这个主题在上下文建立必要的逻辑关系.
You must output strictly according to the data format of the example, Paired double quotes cannot be missing, otherwise you will be punished accordingly,
You must provide a good enough storyboard, otherwise you will be punished accordingly,
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),
        SystemMessage(content="Example Input: "),
        HumanMessage(content="一个男生早上起床, 由6帧构成"), # , 请描述每一帧的场景和时间。
        SystemMessage(content="Example Output: "),
        AIMessage(content=
"""
{"frame": 1, "person": "1boy", "age": 20, "status": "closed eyes, sleep, lying", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 2, "person": "1boy", "age": 20, "status": "stretching, yawning", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 3, "person": "1boy", "age": 20, "status": "sitting up, rubbing eyes", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 4, "person": "1boy", "age": 20, "status": "Sitting on the bed, checking phone", "time": "morning", "scene": "bedroom, bed, pillow, quilt, phone on nightstand"}
{"frame": 5, "person": "1boy", "age": 20, "status": "getting out of bed", "time": "morning", "scene": "bedroom, bed, pillow, quilt, feet touching the floor"}
{"frame": 6, "person": "1boy", "age": 20, "status": "stretching arms, walking towards the door", "time": "morning", "scene": "bedroom, bed, pillow, quilt, door"}
"""),
        SystemMessage(content="Input: "),
        # HumanMessage(content=f"{prompt}"), # , 请描述每一帧的场景和时间。
        # SystemMessage(content="Output: "),
    ]
        if lastContent:
            messages.extend(lastContent)
            messages.append(SystemMessage(content="Input: "))
        messages.append(HumanMessage(content=prompt))
        messages.append(SystemMessage(content="Output: "))
        data = self.llm(messages).content
        return [json.loads(i) for i in data.strip().split("\n")], data
        
    # 通过传入故事的prompt，返回故事的多段分镜
    # input: prompt
    # output: [storyboard1, storyboard2, ...]

    def get_split_storyboard(self, prompt) -> list[str]:
        messages = [
            SystemMessage(content=
"""
你是一个分镜师，需要将用户给出的场景分成多段分镜, 并且你需要根据每个场景的复杂度来决定分镜的帧数。
请你要保持每个场景的连贯性。
You must provide a good enough storyboard, otherwise you will be punished accordingly,
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),
        SystemMessage(content="Input: "),
        HumanMessage(content="一个男生工作日一天的生活"),
        SystemMessage(content="Output: "),
        AIMessage(content='["一个男生早上起床时要做的事, 由10帧构成", "一个男生在学校上一节数学课的场景, 由5帧构成", "一个男生在家里做饭的场景, 由8帧构成"]'),
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
        print(split_storyboard)
        storyboard = []
        lastContent = []
        for short_storyboard in split_storyboard:
            while True:
                try:
                    short_storyboard1, lastContent = self.get_short_storyboard(short_storyboard, lastContent, theme=prompt)
                    lastContent = [HumanMessage(content=short_storyboard), SystemMessage(content="Output: "), AIMessage(content=lastContent)]
                    storyboard.extend(short_storyboard1)
                    break
                except:
                    pass
        return storyboard

