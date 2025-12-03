from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
import sys
import pymupdf
from typing import Literal

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DocumentOCR:
    def __init__(self, model_path: str, device: str = "cuda:0", attn_implementation: str = "auto"):
        """
        初始化 OCR 模型。

        Args:
            model_path (str): 模型路径。
            device (str): 指定使用的设备，如 'cuda:0' 或 'cpu'。
            attn_implementation (str): 注意力实现方式，可选 'flash_attention_2', 'sdpa', 'eager', 或 'auto'。
        """
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 自动检测是否支持 flash_attn
        if attn_implementation == "auto":
            try:
                import flash_attn  # noqa: F401
                self.attn_impl = "flash_attention_2"
            except ImportError:
                print("flash_attention_2 NOT FOUND, using eager instead")
                self.attn_impl = "eager"
        else:
            self.attn_impl = attn_implementation

        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_path,
            _attn_implementation=self.attn_impl,
            trust_remote_code=True,
            use_safetensors=True
        ).eval().to(self.device).to(torch.bfloat16)

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False
    # Gundam: base_size = 1024, image_size = 640, crop_mode = True
    def infer(
            self,
            image_file: str,
            prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
            output_path: str = "./output",
            base_size: int = 1024,
            image_size: int = 640,
            crop_mode: bool = True,
            test_compress: bool = True,
            save_results: bool = True,
    ) -> str:
        """
        执行 OCR 推理。

        Args:
            prompt (str): 输入提示。
            image_file (str): 输入图像路径。
            output_path (str): 输出结果保存路径。
            base_size (int): 基础图像尺寸。
            image_size (int): 输入模型的图像尺寸。
            crop_mode (bool): 是否启用裁剪模式。
            test_compress (bool): 是否启用压缩测试。
            save_results (bool): 是否保存结果到磁盘。

        Returns:
            str: 模型返回的识别结果（文本）。
        """
        result = self.model.infer(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
        )
        return result


def get_image_files(folder: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> list:
    """获取文件夹下所有图片文件"""
    image_files = []
    for file in os.listdir(folder):
        if file.lower().endswith(extensions):
            image_files.append(os.path.join(folder, file))
    return sorted(image_files)  # 排序保证顺序一致


def get_pdf_files(folder: str, extensions: tuple = ('.pdf',)) -> list:
    """获取文件夹下所有PDF文件"""
    pdf_files = []
    for file in os.listdir(folder):
        if file.lower().endswith(extensions):
            pdf_files.append(os.path.join(folder, file))
    return sorted(pdf_files)  # 排序保证顺序一致


def pdf_to_images_pymupdf(pdf_path, output_folder, dpi=300):
    """
    使用 PyMuPDF 将PDF转换为图像
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 打开PDF文件
    pdf_document = pymupdf.open(pdf_path)

    images = []
    for page_num in range(len(pdf_document)):
        # 获取页面
        page = pdf_document.load_page(page_num)

        # 设置转换矩阵（DPI控制）
        zoom = dpi / 72  # 72是PDF的标准DPI
        mat = pymupdf.Matrix(zoom, zoom)

        # 将页面转换为图像（pixmap）
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # 保存图像
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(output_path)
        images.append(pix)

        print(f"已保存: {output_path} (尺寸: {pix.width}x{pix.height})")

    pdf_document.close()
    print(f"转换完成！共 {len(images)} 页")
    return images


def inference(input_folder, output_folder, model, mode: Literal["pdf", "image"]):
    if mode == "pdf":
        try:
            pdf_files = get_pdf_files(folder=input_folder)
        except FileNotFoundError:
            print(f"❌ 错误：输入文件夹不存在 → {input_folder}")
            sys.exit(1)
        except PermissionError:
            print(f"❌ 错误：无权限访问文件夹 → {input_folder}")
            sys.exit(1)
        except NotADirectoryError:
            print(f"❌ 错误：路径不是一个文件夹 → {input_folder}")
            sys.exit(1)
        except OSError as e:
            print(f"❌ 系统错误：无法读取文件夹 '{input_folder}' → {e}")
            sys.exit(1)

        if not pdf_files:
            print(f"⚠️ 警告：文件夹 '{input_folder}' 中未找到PDF）")
            sys.exit(0)

        print(f"Found {len(pdf_files)} PDFs in {input_folder}")
        # 批量处理
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
            # 构造输出路径（例如：output/doc1.txt）
            stem = Path(pdf_path).stem
            pdf_name = stem
            output_path = os.path.join(output_folder, stem)
            pdf_output_path = os.path.join(Path(pdf_path).parent, stem)  # 在pdf同目录下生成同名文件夹以储存转换的图片
            pdf_to_images_pymupdf(pdf_path, pdf_output_path, dpi=300)
            print(output_path)
            try:
                image_files = get_image_files(folder=pdf_output_path)
                print(f"Found {len(image_files)} images in {pdf_output_path}")
            except FileNotFoundError:
                print(f"❌ 错误：输入文件夹不存在 → {input_folder}")
                sys.exit(1)
            except PermissionError:
                print(f"❌ 错误：无权限访问文件夹 → {input_folder}")
                sys.exit(1)
            except NotADirectoryError:
                print(f"❌ 错误：路径不是一个文件夹 → {input_folder}")
                sys.exit(1)
            except OSError as e:
                print(f"❌ 系统错误：无法读取文件夹 '{input_folder}' → {e}")
                sys.exit(1)

            if not image_files:
                print(f"⚠️ 警告：文件夹 '{input_folder}' 中未找到支持的图片等）")
                sys.exit(0)

            print(f"✅ 找到 {len(image_files)} 张图片，开始处理...")
            for n, img_path in enumerate(image_files, 1):
                print(f"[{n}/{len(image_files)}] Processing: {os.path.basename(img_path)}")

                # 构造输出路径（例如：output/doc1.txt）
                stem = Path(img_path).stem
                output_path = os.path.join(output_folder, pdf_name,stem)
                mmd_file = os.path.join(output_path, "result.mmd")
                md_file = os.path.join(output_path, f"result.md")
                print(output_path)
                try:
                    model.infer(
                        image_file=img_path,
                        prompt="<image>\n<|grounding|>Convert the document to markdown.",
                        output_path=output_path,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        test_compress=True,
                        save_results=True,
                    )
                except Exception as e:
                    print(f"❌ Failed on {img_path}: {e}")
                    continue

                if os.path.isfile(mmd_file):
                    # 检查目标文件是否已存在，如果存在则添加数字后缀
                    counter = 1
                    new_md_file = md_file
                    while os.path.exists(new_md_file):
                        name, ext = os.path.splitext(md_file)
                        new_md_file = f"{name}_{counter}{ext}"
                        counter += 1

                    os.rename(mmd_file, new_md_file)

    if mode == "image":
        try:
            image_files = get_image_files(folder=input_folder)
            print(f"Found {len(image_files)} images in {input_folder}")
        except FileNotFoundError:
            print(f"❌ 错误：输入文件夹不存在 → {input_folder}")
            sys.exit(1)
        except PermissionError:
            print(f"❌ 错误：无权限访问文件夹 → {input_folder}")
            sys.exit(1)
        except NotADirectoryError:
            print(f"❌ 错误：路径不是一个文件夹 → {input_folder}")
            sys.exit(1)
        except OSError as e:
            print(f"❌ 系统错误：无法读取文件夹 '{input_folder}' → {e}")
            sys.exit(1)

        if not image_files:
            print(f"⚠️ 警告：文件夹 '{input_folder}' 中未找到支持的图片等）")
            sys.exit(0)

        print(f"✅ 找到 {len(image_files)} 张图片，开始处理...")

        # 批量处理
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(img_path)}")

            # 构造输出路径（例如：output/doc1.txt）
            stem = Path(img_path).stem
            output_path = os.path.join(output_folder, stem)
            mmd_file = os.path.join(output_path, "result.mmd")
            md_file = os.path.join(output_path, f"result.md")
            print(output_path)

            try:
                model.infer(
                    image_file=img_path,
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",
                    output_path=output_path,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    test_compress=True,
                    save_results=True,
                )
            except Exception as e:
                print(f"❌ Failed on {img_path}: {e}")
                continue

            if os.path.isfile(mmd_file):
                # 检查目标文件是否已存在，如果存在则添加数字后缀
                counter = 1
                new_md_file = md_file
                while os.path.exists(new_md_file):
                    name, ext = os.path.splitext(md_file)
                    new_md_file = f"{name}_{counter}{ext}"
                    counter += 1

                os.rename(mmd_file, new_md_file)
