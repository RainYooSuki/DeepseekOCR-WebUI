import module
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSeek OCR')
    parser.add_argument('--input', type=str, default='./input', help='Input folder path')
    parser.add_argument('--output', type=str, default='./output', help='Output folder path')
    parser.add_argument('--model_path', type=str, default='./model', help='Model path')

    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    model_path = args.model_path

    model = module.DocumentOCR(model_path=model_path)
    module.infer(input_folder, output_folder, model)
