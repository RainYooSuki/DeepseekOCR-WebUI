from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
import sys

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


def inference(input_folder, output_folder, model):
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
