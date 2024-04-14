import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import json

class facellm_sci1:
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
You are a storyboard artist and need to provide storyboards based on the scenes given by the user. The number of frames for each scene is up to you, but please maintain logical coherence.
You need to follow the logic and maintain the coherence of each scene, including age, actions, etc.
The 'person' can only be described directly in terms of gender, such as 1boy, 1girl, 1woman, 1man, etc. Use 1boy or 1girl for young age and 1man or 1woman for older age. Do not include information about occupation, etc.
Please continue the story from the (previous storyboard) and write a new storyline, Character settings need to be consistent or logical.
You must provide an (person's age) that conforms to logical changes, otherwise you will be punished!
Each scene will only have one character.
When the age is greater than 10, output multiples of 10, such as 20, 30, 40, 50, 60, etc.
The theme of the user's scene is {theme}, please establish necessary logical relationships based on this theme in the context. For example, age will not change dramatically in a short period of time.
You must output strictly according to the data format of the example. Paired double quotes cannot be missing, otherwise you will be punished accordingly.
You must provide a good enough storyboard, otherwise you will be punished accordingly.
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),
SystemMessage(content="""
Here are some common terms for describing storyboards that you can use as reference.
1. Perspective and Direction:
front view
Profile view / from side
half-front view
Back view
quarter front view
looking at the camera
facing the camera
turned towards the camera
looking away from the camera
facing away from the camera
looking up at the camera
looking down at the camera
looking sideways at the camera

2. Frame Range:
upper body / waist up
Thigh up
knees up
full body
Zoom in on the subject
Zoom out on the subject
Frame the subject tightly
Frame the subject loosely
Center the subject
Move the subject to the left
Move the subject to the right
Move the subject up
Move the subject down
Focus on the subject’s face
Focus on the subject’s body
Blur the background
Isolate the subject
Draw attention to the subject
Avoid cutting off the subject
Isolate the subject’s hands
Isolate the subject’s legs
Isolate the subject’s torso

3. Shot Distance:
Close-up Shot
Medium Shot
Long Shot / wide shots / establishing shots
Extreme Close-up Shot
Extreme Long Shot
cowboy shot
Over the Shoulder Shot
Master Shot
POV Shot
Establishing Shot

4、Positioning, shooting angle：
low-angle shot
Ground Level Shot
Knee Level Shot
Hip Level Shot
Shoulder Level Shot
Eye Level Shot
High-angle Shot
Bird’s-eye View
Aerial Shot
Dutch Camera Angle
"""),
        SystemMessage(content="Example Input: "),
        HumanMessage(content="一个男生早上起床, 由6帧构成"), # , 请描述每一帧的场景和时间。
        SystemMessage(content="Example Output: "),
        AIMessage(content=
"""
{"frame": 1, "person": "1boy", "age": 20, "status": "closed eyes, sleep, lying","screen_description": "High-angle Shot,Medium Shot, front view, upper body", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 2, "person": "1boy", "age": 20, "status": "stretching, yawning","screen_description": "front view,Medium Shot, upper body", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 3, "person": "1boy", "age": 20, "status": "sitting up, rubbing eyes"","screen_description": "front view,Medium Shot, upper body", "time": "morning", "scene": "bedroom, bed, pillow, quilt"}
{"frame": 4, "person": "1boy", "age": 20, "status": "Sitting on the bed, checking phone"","screen_description": "front view,Medium Shot, upper body", "time": "morning", "scene": "bedroom, bed, pillow, quilt, phone on nightstand"}
{"frame": 5, "person": "1boy", "age": 20, "status": "getting out of bed", "time": "morning"","screen_description": "front view,Medium Shot, upper body", "scene": "bedroom, bed, pillow, quilt, feet touching the floor"}
{"frame": 6, "person": "1boy", "age": 20, "status": "stretching arms, walking towards the door"","screen_description": "Long Shot, Back view, full body", "time": "morning", "scene": "bedroom, bed, pillow, quilt, door"}
"""),
        # HumanMessage(content=f"{prompt}"), # , 请描述每一帧的场景和时间。
        # SystemMessage(content="Output: "),
    ]
        if lastContent:
            messages.append(SystemMessage(content=f"previous storyboard: {lastContent}"))
            # messages.append(SystemMessage(content=lastContent))
        else:
            messages.append(SystemMessage(content="previous storyboard: None"))
            # messages.append(SystemMessage(content="None"))
        messages.append(SystemMessage(content="Input: "),)
        messages.append(HumanMessage(content=prompt))
        messages.append(SystemMessage(content="Output: "))
        data = self.llm(messages).content
        return [json.loads(i) for i in data.strip().split("\n")], data
        
    # 通过传入故事的prompt，返回故事的多段分镜
    # input: prompt
    # output: [storyboard1, storyboard2, ...]

    def get_split_storyboard(self, prompt, cnt) -> list[str]:
        messages = [
            SystemMessage(content=
f"""
You are a storyboard artist and need to divide the given scenes into multiple storyboards. You should determine the number of frames for each scene based on its complexity.
Please ensure the coherence of each scene.
The number of frames in each scene needs to be less than 10
You must provide a good enough storyboard, otherwise you will be punished accordingly.
You don't need to output any content outside of the format.
Bonus: You'll get $20 if you get this right.
"""),

        SystemMessage(content="Input: "),
        HumanMessage(content="一个男生工作日一天的生活"),
        SystemMessage(content=f"split into 3 storyboards"),

        SystemMessage(content="Output: "),
        AIMessage(content='["一个男生早上起床时要做的事, 由10帧构成", "一个男生在学校上一节数学课的场景, 由5帧构成", "一个男生在家里做饭的场景, 由8帧构成"]'),
        SystemMessage(content="Input: "),
        HumanMessage(content=prompt),
        SystemMessage(content=f"split into {cnt} storyboards"),
        SystemMessage(content="Output: "),


        ]
        data = self.llm(messages).content
        return eval(data)

    # 最外层的接口，通过传入故事的prompt，返回故事的storyboard
    # input: prompt
    # output: [frame1, frame2, ...]
    
    def get_storyboard_from_prompt(self, prompt, fcnt) -> list:
        # Get the storyboard from the prompt
        split_storyboard = self.get_split_storyboard(prompt, max(fcnt // 100, 1))
        print(split_storyboard)
        storyboard = []
        lastContent = None
        for short_storyboard in split_storyboard:
            while True:
                try:
                    short_storyboard1, lastContent = self.get_short_storyboard(short_storyboard, lastContent, theme=prompt)
                    # lastContent = [HumanMessage(content=short_storyboard), SystemMessage(content="Output: "), AIMessage(content=lastContent)]
                    storyboard.extend(short_storyboard1)
                    break
                except:
                    pass
        return storyboard

