import gradio as gr
import os
from facellm import facellm, facellm_test
from facetest import facellm_sci1
import sys
import shutil

from lorahelper import changelora

from facechain.facechain.utils import snapshot_download, check_ffmpeg, set_spawn_method, project_dir, join_worker_data_dir
# os.environ["http_proxy"] = "http://127.0.0.1:10800"
os.environ["https_proxy"] = "http://127.0.0.1:10800"

# os.chdir('animatediff-cli-prompt-travel')
from animatediff.cli import cli
from llm2sd import change_to_animatediff_prompt, generate_animatediff_config

import json

chinese_to_english = {
    "不指定": "",
    "画面视角": "Viewpoint",
    "正面": "Front view",
    "左侧面": "Profile view / Side view, from left side",
    "右侧面": "Profile view / Side view, from right side",
    "半正面": "Half-front view",
    "背面": "Back view",
    "四分之一正面": "Quarter front view",
    "角度和朝向": "Orientation",
    "看向镜头": "Looking at the camera",
    "面对镜头": "Facing the camera",
    "转向镜头": "Turned towards the camera",
    "不看镜头": "Looking away from the camera",
    "背对镜头": "Facing away from the camera",
    "抬头看向镜头": "Looking up at the camera",
    "低头看向镜头": "Looking down at the camera",
    "向侧面看向镜头": "Looking sideways at the camera",
    "画面范围": "Frame",
    "上半身或腰部以上": "Upper body / Waist up",
    "大腿以上": "Thigh up",
    "膝盖以上": "Knees up",
    "全身": "Full body",
    "镜头远近": "Zoom",
    "放大主题": "Zoom in on the subject",
    "缩小主题": "Zoom out on the subject",
    "紧密地框选主题": "Frame the subject tightly",
    "松散地框选主题": "Frame the subject loosely",
    "将主题置于中心": "Center the subject",
    "将主题移至左侧": "Move the subject to the left",
    "将主题移至右侧": "Move the subject to the right",
    "将主题向上移动": "Move the subject up",
    "将主题向下移动": "Move the subject down",
    "聚焦于主题的脸部": "Focus on the subject’s face",
    "聚焦于主题的身体": "Focus on the subject’s body",
    "模糊背景": "Blur the background",
    "孤立主题，使其与背景分离": "Isolate the subject",
    "吸引注意力至主题上": "Draw attention to the subject",
    "避免裁剪主题": "Avoid cutting off the subject",
    "孤立主题的手部，突出手部动作或姿态": "Isolate the subject’s hands",
    "孤立主题的腿部，突出腿部动作或姿态": "Isolate the subject’s legs",
    "孤立主题的躯干，突出躯干姿势或特征": "Isolate the subject’s torso",
    "机位和拍摄角度": "Camera angle",
    "低角度拍摄": "Low-angle shot",
    "地面级别": "Ground Level Shot",
    "齐膝位": "Knee Level Shot",
    "齐臀位": "Hip Level Shot",
    "齐肩位": "Shoulder Level Shot",
    "齐眼位": "Eye Level Shot",
    "高角度拍摄": "High-angle Shot",
    "鸟瞰视角": "Bird’s-eye View",
    "空中镜头": "Aerial Shot",
    "荷兰式相机角度": "Dutch Camera Angle"
}



 
animateDiff_Model_Path = 'animatediff-cli-prompt-travel/data/models/sd'
animateDiff_Motion_Model_Path = 'animatediff-cli-prompt-travel/data/models/motion-module'
facechain_lora_Model_Path = 'faceoutput'
defalut_n_prompt = "FastNegativeV2, (blurry), (bad-artist:1), (worst quality, low quality:1.4), (bad_prompt_version2:0.8), bad-hands-5, lowres, bad anatomy, bad hands, ((text)), (watermark), error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, ((username)), blurry, (extra limbs), bad-artist-anime, badhandv4, EasyNegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.3, BadDream, (three hands:1.3), (three legs:1.2), (more than two hands:1.4), (more than two legs,:1.4), label, watermark, nsfw, "
SDXL_BASE_MODEL_ID = 'AI-ModelScope/stable-diffusion-xl-base-1.0'

character_model = 'ly261666/cv_portrait_model'
BASE_MODEL_MAP = {
    "leosamsMoonfilm_filmGrain20": "写实模型(Realistic sd_1.5 model)",
    "MajicmixRealistic_v6": "\N{fire}写真模型(Photorealistic sd_1.5 model)",
    "sdxl_1.0": "sdxl_1.0",
}



def update_output_model(uuid):
    folder_list = os.listdir("faceoutput")
    # if not uuid:
    #     if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
    #         raise gr.Error("请登陆后使用! (Please login first)")
    #     else:
    #         uuid = 'qw'
    # folder_list = []
    # for idx, tmp_character_model in enumerate(['AI-ModelScope/stable-diffusion-xl-base-1.0', character_model]):
    #     folder_path = join_worker_data_dir(uuid, tmp_character_model)
    #     if not os.path.exists(folder_path):
    #         continue
    #     else:
    #         files = os.listdir(folder_path)
    #         for file in files:
    #             file_path = os.path.join(folder_path, file)
    #             if os.path.isdir(folder_path):
    #                 file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
    #                 file_lora_path_swift = f"{file_path}/swift"
    #                 if os.path.exists(file_lora_path) or os.path.exists(file_lora_path_swift):
    #                     folder_list.append(file)
    if len(folder_list) == 0:
        return gr.Radio.update(choices=[], value = None)

    return gr.Radio.update(choices=folder_list)

def generate_animatediff_config_click(model, lora_model, openai_api_key, 
                   openai_api_baseurl, prompt, framecnt, width, height, animatediff_motion_model, nprompt, camera_view, ext_prompt,ori_prompt):
    framecnt = int(framecnt)
    width = int(width)
    height = int(height)
    fllm = facellm_sci1(openai_api_key if openai_api_key != "" else "EMPTY", openai_api_baseurl,"gpt-3.5-turbo-16k")

    sb = fllm.get_storyboard_from_prompt(prompt, framecnt)
    extP = f"({chinese_to_english[camera_view]}),({chinese_to_english[ori_prompt]})"

    prompt = change_to_animatediff_prompt(sb, framecnt, ext_prompt=extP, expositive = ext_prompt)
    file,promptres = generate_animatediff_config(prompt,
                                        f"models/sd/{model}", 
                                        lora_model, 
                                        negative_prompt=nprompt, animatediff_motion_model=f"models/motion-module/{animatediff_motion_model}")
    return None, promptres

def generate_image(model, lora_model, openai_api_key, 
                   openai_api_baseurl, prompt, framecnt, width, height, animatediff_motion_model, nprompt, camera_view, ext_prompt, ori_prompt):
    framecnt = int(framecnt)
    width = int(width)
    height = int(height)
    # fllm = facellm_test(openai_api_key if openai_api_key != "" else "EMPTY", "gpt-3.5-turbo-16k", openai_api_baseurl)
    fllm = facellm_sci1(openai_api_key if openai_api_key != "" else "EMPTY", openai_api_baseurl,"gpt-3.5-turbo-16k")
    sb = fllm.get_storyboard_from_prompt(prompt, framecnt)

    extP = f"({chinese_to_english[camera_view]}),({chinese_to_english[ori_prompt]})"
    prompt = change_to_animatediff_prompt(sb, framecnt, ext_prompt=extP, expositive = ext_prompt)
    file, promptres = generate_animatediff_config(prompt,
                                        f"models/sd/{model}", 
                                        lora_model, 
                                        negative_prompt=nprompt, animatediff_motion_model=f"models/motion-module/{animatediff_motion_model}")

    # 调用animateDiff生成图片
    # 修改运行路径到animatediff-cli-prompt-travel

    # 切换到别的conda环境
    # os.system("conda activate animatept")
    # path = os.system(f"python animatediff-cli-prompt-travel/src/animatediff/__main__.py generate -c {file} -W {width} -H {height} -L {framecnt} -C 16")
    path = os.system(f"{sys.executable} animatediff-cli-prompt-travel/src/animatediff/__main__.py generate -c {file} -W {width} -H {height} -L {framecnt} -C 16")
    # cli.command("generate -c  config/prompts/test.json -W 512 -H 512 -L 48 -C 16")
    # cli.invoke(command, param="generate -c  config/prompts/test.json -W 512 -H 512 -L 48 -C 16")
    # print(path)
    outputpath =  f"output/{os.listdir('output')[-1]}"
    # 找到gif结尾的文件
    outputpath1 = os.listdir(f'{outputpath}')
    outputpath = [f"{outputpath}/{file}" for file in outputpath1 if file.endswith('.gif')][0]
    return outputpath, promptres

def train_lora(uuid,
                             base_model_name,
                             instance_images,
                             output_model_name,):
    
    imgfolder = f'./imgs/{output_model_name}/'

    # if not os.path.exists(imgfolder):


    os.makedirs(imgfolder, exist_ok=True)
    # 把图片下载到facechain/facechain/imgs文件夹下
    for file in instance_images:
        shutil.copy(file["name"], imgfolder)

    os.system(f"""accelerate launch facechain/facechain/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=ly261666/cv_portrait_model \
    --revision=v2.0 \
    --sub_path=film/film \
    --dataset_name={imgfolder} \
    --output_dataset_name=./processed \
    --caption_column="text" \
    --resolution=512 --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=200 --checkpointing_steps=5000 \
    --learning_rate=1.5e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=./{facechain_lora_Model_Path} \
    --lora_r=4 --lora_alpha=32 \
    --lora_text_encoder_r=32 --lora_text_encoder_alpha=32""")
    modelpath = f'{facechain_lora_Model_Path}/{output_model_name}.safetensors'
    changelora(f'{facechain_lora_Model_Path}/pytorch_lora_weights.bin', modelpath)
    # 删除原来的文件
    os.remove(f'{facechain_lora_Model_Path}/pytorch_lora_weights.bin')

    #   --resume_from_checkpoint='fromfacecommon'
    # 在这里添加你的Lora训练代码
    return "训练完成"

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

# 定义模型选择框

def train_input():
    # trainer = Trainer()
    
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    output_model_name = gr.Textbox(label="人物lora名称(Character lora name)", value='person1', lines=1)
                    base_model_name = gr.components.Dropdown(choices=['AI-ModelScope/stable-diffusion-v1-5',
                                                           SDXL_BASE_MODEL_ID],
                                                  value='AI-ModelScope/stable-diffusion-v1-5',
                                                  label='基模型')
                    # lora_model = gr.components.Dropdown(choices=['LoraModel1', 'LoraModel2', 'LoraModel3'],value='LoraModel3', label="选择Lora模型")
                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    with gr.Row():
                        upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                        file_count="multiple")
                        #webcam = gr.Button("拍照上传")

                        clear_button = gr.Button("清空图片(Clear photos)")
                    #with gr.Row():
                    #    image = gr.Image(source='webcam',type="filepath",visible=False).style(height=500,width=500)
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                         queue=False)
                    #webcam.click(webcam_image_open,inputs=image,outputs=image)
                    #image.change(add_file_webcam,inputs=[instance_images, image],outputs=instance_images, show_progress=True).then(webcam_image_close,inputs=image,outputs=image)
                    
                    gr.Markdown('''
                        使用说明（Instructions）：
                        ''')
                    gr.Markdown('''
                        - Step 1. 上传计划训练的图片, 1~10张头肩照(注意: 请避免图片中出现多人脸、脸部遮挡等情况, 否则可能导致效果异常)
                        - Step 2. 点击 [开始训练] , 启动形象定制化训练, 每张图片约需1.5分钟, 请耐心等待～
                        - Step 3. 切换至 [形象写真] , 生成你的风格照片<br/><br/>
                        ''')
                    gr.Markdown('''
                        - Step 1. Upload 1-10 headshot photos of yours (Note: avoid photos with multiple faces or face obstruction, which may lead to non-ideal result).
                        - Step 2. Click [Train] to start training for customizing your Digital-Twin, this may take up-to 1.5 mins per image.
                        - Step 3. Switch to [Portrait] Tab to generate stylized photos.
                        ''')

        run_button = gr.Button('开始训练(等待上传图片加载显示出来再点, 否则会报错)... '
                               'Start training (please wait until photo(s) fully uploaded, otherwise it may result in training failure)')

        with gr.Box():
            gr.Markdown('''
            <center>请等待训练完成，请勿刷新或关闭页面。</center>

            <center>(Please wait for the training to complete, do not refresh or close the page.)</center>
            ''')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验。

            (If you are experiencing prolonged waiting time, you may try on [ModelScope NoteBook](https://modelscope.cn/my/mynotebook/preset) to prepare your dedicated environment.)

            安装方法请参考：https://github.com/modelscope/facechain .

            (You may refer to: https://github.com/modelscope/facechain for installation instruction.)
            ''')

        run_button.click(fn=train_lora,
                         inputs=[
                             uuid,
                             base_model_name,
                             instance_images,
                             output_model_name,
                         ],
                         outputs=[output_message])

    return demo


def generate_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    openai_api_key = gr.Textbox(label="OpenAI API Key", lines=1)
                    openai_api_baseurl = gr.Textbox(label="OpenAI API Base URL", value="http://localhost:8000/v1" , lines=1)

                    animateDiff_Models = [file for file in os.listdir(animateDiff_Model_Path) if file.endswith('.bin') or file.endswith('.pt') or file.endswith('.pth') or file.endswith('.safetensors') or file.endswith('.ckpt') ]
                    update_button = gr.Button("更新模型(Update models)")
                    lora_models = [file for file in os.listdir(facechain_lora_Model_Path) if file.endswith('.bin') or file.endswith('.pt') or file.endswith('.pth') or file.endswith('.safetensors') or file.endswith('.ckpt')]
                    lora_models.append("")
                    animateDiff_Motion_Models = [file for file in os.listdir(animateDiff_Motion_Model_Path) if file.endswith('.bin') or file.endswith('.pt') or file.endswith('.pth') or file.endswith('.safetensors') or file.endswith('.ckpt')]
                    model = gr.components.Dropdown(choices=animateDiff_Models, label="选择模型")
                    animatediff_motion_model = gr.components.Dropdown(choices=animateDiff_Motion_Models, label="选择Animatediff模型")
                    lora_model = gr.components.Dropdown(choices=lora_models, label="选择Lora模型")
                    prompt = gr.components.Textbox(lines=2, placeholder='在这里输入prompt...', label="输入Prompt")
                    with gr.Row():
                        width = gr.Number(label="图片宽度(Width)", default=512, min=1, max=1000, step=1, value=512)
                        height = gr.Number(label="图片高度(Height)", default=512, min=1, max=1000, step=1, value=512)
                        framecnt = gr.Number(label="帧数(Frame count)", default=100, min=1, max=1000, step=1, value=300)
                    ex_positives = gr.Textbox(label ="额外提示(Extra prompt)")
                    
                    with gr.Row():
                        # 用于不同种风格的单选按钮
                        camera_view = gr.Radio(label="选择镜头(Select model)", choices=["不指定","背面", "正面", "左侧面", "右侧面", "半正面", "四分之一正面"], value="不指定")
                    
                    with gr.Row():
                        Orientation = gr.Radio(label="选择朝向(Select Orientation)", choices=["不指定", "看向镜头", "面对镜头", "转向镜头", "不看镜头", "背对镜头", "抬头看向镜头", "低头看向镜头", "向侧面看向镜头"], value="不指定")
                    
                    

                    with gr.Accordion("高级选项(Advanced options)", default_open=False):
                        nprompt = gr.Textbox(label="负向提示词(Advanced options)", lines=1, value=defalut_n_prompt)

                    with gr.Row():
                        generate_config = gr.Button("生成配置文件(Generate config file)")
                    with gr.Row():
                        run_button = gr.Button("生成图片", )
                    
                    outconfig = gr.components.Textbox(label="生成的配置文件(Generated config file)")
                    # image = gr.components.Textbox(label="生成的配置文件(Generated config file)")
                    image = gr.outputs.Image(type="filepath", label="生成的图片")

        update_button.click(fn=update_output_model, inputs=[uuid], outputs=[lora_model], queue=False)
                    # output_message = gr.Label()
                    # gr.Markdown('''
                    #     使用说明（Instructions）：
                    #     ''')
                    # gr.Markdown('''
                    #     - Step 1. 选择模型和Lora模型
                    #     - Step 2. 输入Prompt
                    #     - Step 3. 点击 [生成图片] , 生成你的风格照片<br/><br/>
                    #     ''')
                    # gr.Markdown('''
                    #     - Step 1. Select model and Lora model
                    #     - Step 2. Enter Prompt
                    #     - Step 3. Click [Generate Image] to generate your stylized photo
                        # ''')
        run_button.click(fn=generate_image,
                         inputs=[
                             model,
                             lora_model,
                             openai_api_key,
                             openai_api_baseurl,
                             prompt,
                             framecnt,
                             width, height, animatediff_motion_model, nprompt, camera_view, ex_positives, Orientation
                         ],
                         outputs=[image, outconfig])
        generate_config.click(fn=generate_animatediff_config_click, inputs=[
                             model,
                             lora_model,
                             openai_api_key,
                             openai_api_baseurl,
                             prompt,
                             framecnt,
                             width, height, animatediff_motion_model, nprompt, camera_view, ex_positives, Orientation
                         ], outputs=[image,outconfig])
    pass


# model = gr.components.Dropdown(choices=['Model1', 'Model2', 'Model3'], label="选择模型")

# # 定义lora模型选择框
# lora_model = gr.components.Dropdown(choices=['LoraModel1', 'LoraModel2', 'LoraModel3'], label="选择Lora模型")

# # 定义文本框输入prompt
# prompt = gr.components.Textbox(lines=2, placeholder='在这里输入prompt...', label="输入Prompt")

# # 定义图片显示区域
# image = gr.outputs.Image(type="filepath", label="生成的图片")


# upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
#                                                         file_count="multiple")

# iface1 = gr.Interface(fn=train_lora, inputs=upload_button, outputs="text", title="Lora训练页面")


# iface2 = gr.Interface(fn=generate_image, inputs=[model, lora_model, prompt], outputs=image)
# # iface = gr.Interface(pages=[["lora训练", iface1], ["Page 2", iface2]])
# iface = gr.TabbedInterface([iface1, iface2], ["Hello World", "Bye World"])


with gr.Blocks(css='style.css') as demo:
    with gr.Tabs():
        with gr.Tab("Lora训练"):
            train_input()
        with gr.Tab("生成视频"):
            generate_input()
        with gr.Tab("关于"):
            gr.Label("这是一个关于页面")
demo.launch()
# iface.launch()