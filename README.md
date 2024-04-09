# FacechainLearn

## Installation

1. 构建conda虚拟环境

```shell
conda create -n animatediffgo python=3.10
conda activate animatediffgo
```

2. 安装pytroch

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

3. 安装其他依赖

```shell
pip install -r requirements.txt
```

4. 安装Facechain
请参照facechain的安装文档安装facechain

5. 安装animatediff-cli-prompt-travel
请参照animatediff-cli-prompt-travel的安装文档安装animatediff-cli-prompt-travel, 其中pytorch无需安装


## Usage

1. 如果您有openai的API key, 则无需其他准备，直接运行
2. 如果您没有openai的API key, 或是希望使用自己本地的模型, 可以预先启动本地的模型openai接口, 推荐使用chatglm3
3. 在本目录下运行以下命令
```shell
python app.py
```

## model

需要提前准备好sd模型和animatediff模型  
sd模型路径: animatediff-cli-prompt-travel/data/models/sd  
animatediff模型路径: animatediff-cli-prompt-travel/data/models/motion-module