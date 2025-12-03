import os
import subprocess
import sys
import time
import gradio as gr

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_single_image(image):
    """
    处理单张上传的图片
    """
    if image is None:
        return "请上传一张图片", None

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
    image.save(image_path)

    # 运行main.py进行推理
    try:
        # 使用encoding='utf-8'和errors='ignore'来避免编码问题
        result = subprocess.run([sys.executable, "main.py", "--input", "./temp", "--output", "./output"],
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


def process_batch(input_folder, output_folder):
    """
    批量处理图片
    """
    # 运行main.py进行批量推理
    try:
        # 使用encoding='utf-8'和errors='ignore'来避免编码问题
        result = subprocess.run([sys.executable, "main.py", "--input", input_folder, "--output", output_folder],
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
    gr.Markdown("# DeepSeek OCR 文档识别工具")
    gr.Markdown("使用DeepSeek OCR模型将图片中的文档转换为Markdown格式")

    with gr.Tab("单张图片处理"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")
                run_button = gr.Button("开始OCR识别", variant="primary")

            with gr.Column():
                markdown_output = gr.Textbox(label="Markdown结果", lines=15)
                image_output = gr.Image(label="结果图片", interactive=False)

        run_button.click(
            fn=process_single_image,
            inputs=[image_input],
            outputs=[markdown_output, image_output]
        )

    with gr.Tab("批量处理"):
        with gr.Row():
            with gr.Column():
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
                output_folder_input
            ],
            outputs=batch_output
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)