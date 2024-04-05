import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def GetStableDiffusionMessages(input_text):
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
        HumanMessage(content=f"{input_text}, 由5帧构成"), # , 请描述每一帧的场景和时间。
        SystemMessage(content="Output: "),
    ]

    return messages


llm = ChatOpenAI(
        model_name="chatglm",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",
        streaming=False,
    )
print(llm(GetStableDiffusionMessages("一个男生在学校上一节数学课的场景")).content)