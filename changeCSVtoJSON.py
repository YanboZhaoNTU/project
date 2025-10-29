import csv
import json
from collections import defaultdict
import ast

# 读取CSV文件并按video_id分组
videos_data = {}

with open('video_mme_test.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        video_id = row['video_id']
        
        # 如果是新的video_id，创建新的视频条目
        if video_id not in videos_data:
            videos_data[video_id] = {
                "video_id": video_id,
                "duration": row['duration'],
                "domain": row['domain'],
                "sub_category": row['sub_category'],
                "url": row['url'],
                "videoID": row['videoID'],
                "questions": []
            }
        
        # 处理options列表
        try:
            # 尝试解析options_list字符串为列表
            options_str = row['options_list']
            # 按照"A. "、"B. "等分隔选项
            options = []
            for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
                if f'{letter}. ' in options_str:
                    start_idx = options_str.index(f'{letter}. ')
                    # 找到下一个字母的位置或字符串结束
                    next_letter = chr(ord(letter) + 1)
                    if f'{next_letter}. ' in options_str:
                        end_idx = options_str.index(f'{next_letter}. ')
                        option_text = options_str[start_idx:end_idx].strip()
                    else:
                        option_text = options_str[start_idx:].strip()
                    options.append(option_text)
                else:
                    break
        except:
            # 如果解析失败，使用原始字符串
            options = [row['options_list']]
        
        # 构建response字段 (answer + 对应的选项文本)
        answer = row['answer']
        response = answer
        try:
            answer_idx = int(row['answer_idx'])
            if answer_idx < len(options):
                response = options[answer_idx]
        except:
            pass
        
        # 添加问题到questions数组
        question_data = {
            "question_id": row['question_id'],
            "task_type": row['task_type'],
            "question": row['question'],
            "options": options,
            "answer": answer,
            "response": response
        }
        
        videos_data[video_id]['questions'].append(question_data)

# 转换为列表
videos_list = list(videos_data.values())

# 保存为JSON文件
with open('video_mme_test.json', 'w', encoding='utf-8') as f:
    json.dump(videos_list, f, ensure_ascii=False, indent=4)

print(f"已成功处理 {len(videos_list)} 个视频")
print(f"数据已保存到 video_mme_test.json")

# 打印前两个视频的示例
print("\n=== 前2个视频的数据示例 ===\n")
print(json.dumps(videos_list[:2], ensure_ascii=False, indent=4))
