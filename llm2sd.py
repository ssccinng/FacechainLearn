import argparse
import json

def read_jsonl_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Process the data as needed
            data_list.append(data)
    return data_list


def change_to_animatediff_prompt(storyboardlist, max_frame = 100):
    # Convert the storyboard to the format required by the model
    # framecnt = 0
    res = {}
    framecnt = len(storyboardlist)
    idx = 0
    for storyboard in storyboardlist:
        res[idx * max_frame // framecnt] = f'{storyboard.get("person", "1boy")},{storyboard.get("age", "")} year old,{storyboard.get("time", "")},{storyboard.get("status", "")},{storyboard.get("scene", "")}'
        idx += 1
        # print(f'"{framecnt}" : "a handsome man,{storyboard["person"]},{storyboard["age"]} year old,{storyboard["time"]},{storyboard["action"]},{storyboard["scene"]}",')
        # framecnt += 10   
        # Process the storyboard as needed
    return res

def generate_animatediff_config(storyboard, model = None, lora_model = None, negative_prompt = None):
    configTemplate = json.load(open('config_template/template.json'))

    configTemplate['prompt_map'] = storyboard

    if model:
        configTemplate['path'] = model

    if lora_model:
        configTemplate['prompt']['lora_map'] = {f'../../faceoutput/{lora_model}': 1.0}
    
    if negative_prompt:
        configTemplate['prompt']['n_prompt'] = negative_prompt

    # 获取当前时间作为文件名
    import time
    filename = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = f'out_config/config{filename}.json'
    json.dump(configTemplate, open(filename, 'w'), indent=4)
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read JSONL file')
    parser.add_argument('file', help='Path to the JSONL file')
    args = parser.parse_args()

    storyboardlist = read_jsonl_file(args.file)
    change_to_animatediff_prompt(storyboardlist)


