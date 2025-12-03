import os
import subprocess
import sys
import time
import gradio as gr
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_single_image(image, mode):
    """
    处理单张上传的图片或PDF
    """
    # 初始化pdf_filename变量
    pdf_filename = None
    
    if mode == "image":
        if image is None:
            return "请上传一张图片", None

        # 检查上传的文件是否为图片
        if isinstance(image, dict) and 'name' in image:
            # 处理文件上传的情况
            if not image['name'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                return "请上传有效的图片文件 (png, jpg, jpeg, bmp, tiff)", None
            
            # 创建临时输入目录
            temp_input_dir = "./temp"
            os.makedirs(temp_input_dir, exist_ok=True)

            # 清空临时输入目录
            for file in os.listdir(temp_input_dir):
                file_path = os.path.join(temp_input_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # 保存上传的图片到temp目录
            timestamp = int(time.time())
            image_path = os.path.join(temp_input_dir, f"uploaded_image_{timestamp}.jpg")
            
            # 复制上传的图片文件
            with open(image_path, "wb") as f:
                with open(image['name'], "rb") as uploaded_file:
                    f.write(uploaded_file.read())
        elif isinstance(image, Image.Image):
            # 处理直接绘制图像的情况
            # 创建临时输入目录
            temp_input_dir = "./temp"
            os.makedirs(temp_input_dir, exist_ok=True)

            # 清空临时输入目录
            for file in os.listdir(temp_input_dir):
                file_path = os.path.join(temp_input_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # 保存绘制的图片到temp目录
            timestamp = int(time.time())
            image_path = os.path.join(temp_input_dir, f"uploaded_image_{timestamp}.jpg")
            image.save(image_path)
        elif hasattr(image, 'name'):  # 处理其他可能的文件对象
            # 检查文件扩展名
            if not image.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                return "请上传有效的图片文件 (png, jpg, jpeg, bmp, tiff)", None
                
            # 创建临时输入目录
            temp_input_dir = "./temp"
            os.makedirs(temp_input_dir, exist_ok=True)

            # 清空临时输入目录
            for file in os.listdir(temp_input_dir):
                file_path = os.path.join(temp_input_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # 保存上传的图片到temp目录
            timestamp = int(time.time())
            image_path = os.path.join(temp_input_dir, f"uploaded_image_{timestamp}.jpg")
            
            # 复制上传的图片文件
            with open(image_path, "wb") as f:
                with open(image.name, "rb") as uploaded_file:
                    f.write(uploaded_file.read())
        else:
            return "请上传一张图片", None

    elif mode == "pdf":
        if image is None:
            return "请上传一个PDF文件", None

        # 检查上传的文件是否为PDF
        if isinstance(image, dict) and 'name' in image:
            if not image['name'].lower().endswith('.pdf'):
                return "请上传有效的PDF文件", None
        elif hasattr(image, 'name'):
            if not image.name.lower().endswith('.pdf'):
                return "请上传有效的PDF文件", None
        else:
            return "请上传有效的PDF文件", None

        # 创建临时输入目录
        temp_input_dir = "./temp"
        os.makedirs(temp_input_dir, exist_ok=True)

        # 清空临时输入目录
        for file in os.listdir(temp_input_dir):
            file_path = os.path.join(temp_input_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # 保存上传的PDF到temp目录
        timestamp = int(time.time())
        pdf_filename = f"uploaded_document_{timestamp}.pdf"
        pdf_path = os.path.join(temp_input_dir, pdf_filename)
        
        # 复制上传的PDF文件
        if isinstance(image, dict) and 'name' in image:
            with open(pdf_path, "wb") as f:
                with open(image['name'], "rb") as uploaded_file:
                    f.write(uploaded_file.read())
        elif hasattr(image, 'name'):
            with open(pdf_path, "wb") as f:
                with open(image.name, "rb") as uploaded_file:
                    f.write(uploaded_file.read())

    # 运行main.py进行推理
    try:
        # 使用encoding='utf-8'和errors='ignore'来避免编码问题
        result = subprocess.run([sys.executable, "main.py", "--input", "./temp", "--output", "./output", "--mode", mode],
                                capture_output=True,
                                timeout=300,
                                encoding='utf-8',
                                errors='ignore')
        if result.returncode != 0:
            return f"推理过程出错: {result.stderr}", None
    except subprocess.TimeoutExpired:
        return "推理超时", None
    except Exception as e:
        return f"运行推理时出错: {str(e)}", None

    # 查找输出结果
    output_dir = "./output"
    if not os.path.exists(output_dir):
        return "未找到输出目录", None

    if mode == "image":
        # 查找最新创建的文件夹
        subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs:
            return "未找到输出结果", None

        latest_dir = max(subdirs, key=os.path.getctime)

        # 查找生成的markdown文件
        md_file = os.path.join(latest_dir, "result.md")
        if not os.path.exists(md_file):
            mmd_file = os.path.join(latest_dir, "result.mmd")
            if os.path.exists(mmd_file):
                # 重命名为md文件
                os.rename(mmd_file, md_file)

        # 如果存在md文件，读取内容
        if os.path.exists(md_file):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(md_file, 'r', encoding='gbk') as f:
                        markdown_content = f.read()
                except UnicodeDecodeError:
                    with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                        markdown_content = f.read()
        else:
            markdown_content = "未能生成Markdown文件"

        # 查找生成的图片文件用于展示
        result_image = None
        for file in os.listdir(latest_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                result_image = os.path.join(latest_dir, file)
                break

        return markdown_content, result_image

    elif mode == "pdf" and pdf_filename:
        # 对于PDF模式，返回PDF文件名对应的输出目录路径
        pdf_name = Path(pdf_filename).stem
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        
        if os.path.exists(pdf_output_dir):
            message = f"PDF处理完成，结果保存在目录: {pdf_output_dir}\n\n"
            
            # 尝试读取该目录下的所有页面结果
            page_dirs = [d for d in os.listdir(pdf_output_dir) 
                         if os.path.isdir(os.path.join(pdf_output_dir, d))]
            
            for page_dir in sorted(page_dirs):
                page_path = os.path.join(pdf_output_dir, page_dir)
                md_file = os.path.join(page_path, "result.md")
                
                if os.path.exists(md_file):
                    message += f"\n--- 页面 {page_dir} ---\n"
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            message += f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(md_file, 'r', encoding='gbk') as f:
                                message += f.read()
                        except UnicodeDecodeError:
                            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                                message += f.read()
                else:
                    message += f"\n页面 {page_dir} 未生成Markdown结果\n"
            
            return message, None
        else:
            return f"未找到PDF处理结果，预期目录: {pdf_output_dir}", None
    else:
        return "处理过程中发生未知错误", None


def process_batch(input_folder, output_folder, mode):
    """
    批量处理图片或PDF
    """
    # 运行main.py进行批量推理
    try:
        # 使用encoding='utf-8'和errors='ignore'来避免编码问题
        result = subprocess.run([sys.executable, "main.py", "--input", input_folder, "--output", output_folder, "--mode", mode],
                                capture_output=True,
                                timeout=600,
                                encoding='utf-8',
                                errors='ignore')
        if result.returncode != 0:
            return f"批量推理过程出错: {result.stderr}"

        return result.stdout if result.stdout else "批量处理完成"
    except subprocess.TimeoutExpired:
        return "批量推理超时"
    except Exception as e:
        return f"运行批量推理时出错: {str(e)}"


with gr.Blocks(title="DeepSeek OCR WebUI") as demo:
    gr.Markdown("# DeepSeek OCR WebUI")

    with gr.Tab("单张图片处理"):
        with gr.Row():
            with gr.Column():
                mode_selector = gr.Radio(
                    choices=["image", "pdf"],
                    value="image",
                    label="选择处理模式"
                )
                image_input = gr.File(label="上传文件", file_types=["image", ".pdf"])
                run_button = gr.Button("开始OCR识别", variant="primary")

            with gr.Column():
                markdown_output = gr.Textbox(label="Markdown结果", lines=15)
                image_output = gr.Image(label="结果图片", interactive=False)

        run_button.click(
            fn=process_single_image,
            inputs=[image_input, mode_selector],
            outputs=[markdown_output, image_output]
        )

    with gr.Tab("批量处理"):
        with gr.Row():
            with gr.Column():
                batch_mode_selector = gr.Radio(
                    choices=["image", "pdf"],
                    value="image",
                    label="选择处理模式"
                )
                input_folder_input = gr.Textbox(
                    label="输入文件夹路径",
                    value="./input"
                )
                output_folder_input = gr.Textbox(
                    label="输出文件夹路径",
                    value="./output"
                )
                batch_run_button = gr.Button("开始批量处理", variant="primary")

            with gr.Column():
                batch_output = gr.Textbox(label="处理结果", lines=15)

        batch_run_button.click(
            fn=process_batch,
            inputs=[
                input_folder_input,
                output_folder_input,
                batch_mode_selector
            ],
            outputs=batch_output
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)