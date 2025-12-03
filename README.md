# DeepSeek OCR WebUI

基于DeepSeek OCR模型的文档识别工具，可将图片中的文档内容转换为Markdown格式。

## 功能特性

- 单张图片OCR识别
- 批量图片处理
- 图形化Web界面
- 支持GPU加速推理
- 自动生成Markdown格式文档
- 边界框可视化结果


## 安装步骤

1. 克隆项目代码：
```bash
git clone <repository-url>
cd DeepseekOCR-WebUI
```

2. 安装依赖：
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

3. 下载模型文件并放置在 `model` 文件夹中

## 目录结构

```
DeepseekOCR-WebUI/
├── model/                 # 模型文件目录
├── input/                 # 输入图片目录（批量处理用）
├── output/                # 输出结果目录
├── main.py               # 命令行主程序
├── webui.py              # 图形化界面程序
├── module.py             # 模型封装模块
└── requirements.txt      # 依赖包列表
```

## 使用方法

### 启动Web界面

```bash
python webui.py
```

然后在浏览器中打开 `http://localhost:7860`

#### 单张图片处理
1. 在Web界面中切换到"单张图片处理"标签页
2. 点击"上传图片"选择要识别的图片
3. 点击"开始OCR识别"按钮
4. 等待处理完成后，结果将显示在右侧

#### 批量处理
1. 在Web界面中切换到"批量处理"标签页
2. 设置输入文件夹路径（默认为`./input`）
3. 设置输出文件夹路径（默认为`./output`）
4. 点击"开始批量处理"按钮

### 命令行使用

```bash
python main.py
```

## 许可证

本项目采用MIT许可证，详情请见LICENSE文件。