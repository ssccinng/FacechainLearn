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

def change_to_animatediff_prompt(storyboardlist):
    # Convert the storyboard to the format required by the model
    framecnt = 0
    for storyboard in storyboardlist:
        print(f'"{framecnt}" : "a handsome man,{storyboard["person"]},{storyboard["age"]} year old,{storyboard["time"]},{storyboard["action"]},{storyboard["scene"]}",')
        framecnt += 4   
        # Process the storyboard as needed



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read JSONL file')
    parser.add_argument('file', help='Path to the JSONL file')
    args = parser.parse_args()

    storyboardlist = read_jsonl_file(args.file)
    change_to_animatediff_prompt(storyboardlist)


