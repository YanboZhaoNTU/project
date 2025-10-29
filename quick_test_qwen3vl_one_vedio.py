"""
快速测试脚本 - 用于验证 Qwen3-VL 环境配置
测试单个视频，快速检查模型是否能正常工作
"""

import json
import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def quick_test():
    """快速测试单个视频"""
    
    print("="*50)
    print("Qwen3-VL-8B-Instruct 快速测试")
    print("="*50)
    
    # 配置
    VIDEO_DIR = Path("Video-MME/unzipped/data")
    JSON_PATH = "video_mme_test.json"
    
    # 1. 加载测试数据
    print("\n[1/4] 加载测试数据...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"✓ 成功加载 {len(test_data)} 个视频的测试数据")
        
        # 选择第一个视频
        first_video = test_data[0]
        video_id = first_video['videoID']
        print(f"✓ 测试视频: {video_id}")
    except Exception as e:
        print(f"✗ 加载测试数据失败: {e}")
        return
    
    # 2. 检查视频文件
    print("\n[2/4] 检查视频文件...")
    video_path = None
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
    for ext in extensions:
        test_path = VIDEO_DIR / f"{video_id}{ext}"
        if test_path.exists():
            video_path = str(test_path)
            print(f"✓ 找到视频文件: {video_path}")
            break
    
    if video_path is None:
        print(f"✗ 未找到视频文件 {video_id}")
        print(f"  请检查视频是否在目录: {VIDEO_DIR}")
        return
    
    # 3. 加载模型
    print("\n[3/4] 加载 Qwen3-VL-8B-Instruct 模型...")
    print("  (首次运行需要下载模型，可能需要几分钟...)")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        print("✓ 模型加载成功!")
        print(f"  设备: {model.device}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 4. 测试第一个问题
    print("\n[4/4] 测试模型推理...")
    try:
        question_data = first_video['questions'][0]
        question = question_data['question']
        options = question_data['options']
        true_answer = question_data['answer']
        
        print(f"\n问题: {question}")
        print("选项:")
        for opt in options:
            print(f"  {opt}")
        print(f"正确答案: {true_answer}")
        
        # 构建提示
        prompt = f"{question}\n\n"
        for option in options:
            prompt += f"{option}\n"
        prompt += "\nPlease select the correct answer (A, B, C, or D)."
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 准备输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # 生成回答
        print("\n正在生成回答...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\n模型回答: {response}")
        
        # 提取答案
        import re
        match = re.search(r'\b([ABCD])\b', response.upper())
        predicted_answer = match.group(1) if match else None
        
        print(f"预测答案: {predicted_answer}")
        print(f"是否正确: {'✓ 正确' if predicted_answer == true_answer else '✗ 错误'}")
        
        print("\n" + "="*50)
        print("✓ 快速测试完成! 环境配置正常")
        print("="*50)
        print("\n可以运行完整测试:")
        print("  python test_qwen3_vl.py")
        
    except Exception as e:
        print(f"✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    quick_test()