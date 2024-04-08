from facellm import facellm_test

from llm2sd import change_to_animatediff_prompt, generate_animatediff_config

facellmt = facellm_test(api_key="EMPTY", openai_api_base="http://localhost:8000/v1", model_name="chatglm")
sb = facellmt.get_storyboard_from_prompt("一个游戏主播的成长履历")
# sb1 = facellmt.get_short_storyboard("一个游戏主播的成长履历, 由30帧构成")
print(sb)


prompt = change_to_animatediff_prompt(sb, 300)
print(generate_animatediff_config(prompt))