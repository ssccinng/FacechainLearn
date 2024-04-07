from facellm import facellm_test

from llm2sd import change_to_animatediff_prompt, generate_animatediff_config

facellmt = facellm_test(api_key="EMPTY", openai_api_base="http://localhost:8000/v1", model_name="chatglm")
sb = facellmt.get_storyboard_from_prompt("一个女生假日出门购物的场景")
print(sb)


prompt = change_to_animatediff_prompt(sb, 300)
print(generate_animatediff_config(prompt))