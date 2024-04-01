import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
model_path = "F:\\GLM-facechain\\ChatGLM3\\chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

input_text = """
你是一个分镜师，需要给提示词给stable diffusion生成分镜，你需要把角色和动态消息都设置好，你需要用英文来描述这些信息。你只需要描述人物场景和时间就行。人物是1girl或者1boy。
"请描述一个男生的早晨，他很不情愿的从梦中醒来，做完早上该做的事情去上班，格式按照下面这些来:
{'frame': 1, 'person': '1girl', 'action': 'wakes up beside a window, looking at the pre-dawn street.', 'time': 'early morning'}
{'frame': 2, 'person': '1girl', 'action': 'walks through a hallway with rock band posters.', 'time': 'morning'}
{'frame': 3, 'person': '1girl', 'action': 'brushes her teeth in the bathroom.', 'time': 'morning'}
{'frame': 4, 'person': '1girl', 'action': 'steps out the door, sleepy.', 'time': 'morning'}
{'frame': 5, 'person': '1girl', 'action': 'girl is looking around.', 'time': 'morning'}
请生成最正确的格式，一共生成十条。
"""

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

output = model.generate(input_ids, max_length=1000, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
# import torch

# print(torch.cuda.is_available())
