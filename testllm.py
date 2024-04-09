from facellm import facellm_test
from facetest import facellm_sci1

from llm2sd import change_to_animatediff_prompt, generate_animatediff_config

facellmt = facellm_sci1(api_key="EMPTY", openai_api_base="http://localhost:8000/v1", model_name="chatglm")
sb = facellmt.get_storyboard_from_prompt("一个游戏主播的成长履历")
# sb1 = facellmt.get_short_storyboard("一个游戏主播的成长履历, 由30帧构成")
print(sb)
# cc = """
# [{'frame': 1, 'person': '1girl', 'age': 22, 'status': 'lying down, closed eyes, sleep, holding phone', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone'}, {'frame': 2, 'person': '1girl', 'age': 22, 'status': 'opening eyes, stretching', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone'}, {'frame': 3, 'person': '1girl', 'age': 22, 'status': 'getting out of bed, putting on shoes', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, shoes on'}, {'frame': 4, 'person': '1girl', 'age': 22, 'status': 'walking towards the bathroom, carrying toothbrush and hair brush', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, bathroom door'}, {'frame': 5, 'person': '1girl', 'age': 22, 'status': 'brushing teeth with toothbrush', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, bathroom sink'}, {'frame': 6, 'person': '1girl', 'age': 22, 'status': 'washing face with water from bathroom sink', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, bathroom sink'}, {'frame': 7, 'person': '1girl', 'age': 22, 'status': 'dressing, putting on clothes', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, clothes on'}, {'frame': 8, 'person': '1girl', 'age': 22, 'status': 'leaving bedroom, closing door', 'time': 'morning', 'scene': 'bedroom, bed, alarm clock, phone, hallway'}, {'frame': 9, 'person': '1girl', 'age': 22, 'status': 'going to kitchen, carrying coffee mug', 'time': 'morning', 'scene': 'kitchen, counter, coffee machine, coffee mug'}, {'frame': 10, 'person': '1girl', 'age': 22, 'status': 'drinking coffee while talking on phone in living room', 'time': 'morning', 'scene': 'living room, couch, armchair, phone, coffee table'}, {'frame': 1, 'person': '1girl', 'age': 25, 'status': 'standing in front of mirror, holding makeup bag', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 2, 'person': '1girl', 'age': 25, 'status': 'applying foundation, blush, and highlighter', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 3, 'person': '1girl', 'age': 25, 'status': 'brushing eyelashes, applying mascara', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 4, 'person': '1girl', 'age': 25, 'status': 'applying lipstick and chapstick', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 5, 'person': '1girl', 'age': 25, 'status': 'finishing touch, looking at reflection', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 6, 'person': '1girl', 'age': 25, 'status': 'smiling at herself in the mirror, adjusting hair and makeup', 'time': 'afternoon', 'scene': 'bathroom, mirror, vanity, makeup bag'}, {'frame': 1, 'person': '1girl', 'age': 28, 'status': 'sitting at table, holding fork and knife', 'time': 'morning', 'scene': 'kitchen or dining room, breakfast table, plates, utensils'}, {'frame': 2, 'person': '1girl', 'age': 28, 'status': 'putting food into her mouth, chewing and swallowing', 'time': 'morning', 'scene': 'kitchen or dining room, breakfast table, plates, utensils'}, {'frame': 3, 'person': '1girl', 'age': 28, 'status': 'drinking from a glass of orange juice', 'time': 'morning', 'scene': 'kitchen or dining room, breakfast table, plates, utensils'}, {'frame': 4, 'person': '1girl', 'age': 28, 'status': 'leaning back in chair, smiling and enjoying breakfast', 'time': 'morning', 'scene': 'kitchen or dining room, breakfast table, plates, utensils'}, {'frame': 1, 'person': '1girl', 'age': 28, 'status': 'walking down the street, carrying a bag', 'time': 'morning', 'scene': 'outside, street, cityscape'}, {'frame': 2, 'person': '1girl', 'age': 28, 'status': 'checking phone for notifications', 'time': 'morning', 'scene': 'outside, street, cityscape'}, {'frame': 3, 'person': '1girl', 'age': 28, 'status': 'crossing the street, looking both ways', 'time': 'morning', 'scene': 'outside, street, cityscape'}, {'frame': 4, 'person': '1girl', 'age': 28, 'status': 'entering an office building, holding the door open for others', 'time': 'morning', 'scene': 'office building, entrance, doorman'}, {'frame': 5, 'person': '1girl', 'age': 28, 'status': 'waving goodbye to friends in the elevator', 'time': 'morning', 'scene': 'office building, elevator'}, {'frame': 6, 'person': '1girl', 'age': 28, 'status': 'seating herself at her desk, turning on computer', 'time': 'morning', 'scene': 'office building, desk'}, {'frame': 7, 'person': '1girl', 'age': 28, 'status': 'smiling and greeting colleagues, starting work', 'time': 'morning', 'scene': 'office building, workspace'}]
# """

prompt = change_to_animatediff_prompt(sb, 300)
print(generate_animatediff_config(prompt))