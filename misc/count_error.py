import json
import re
import matplotlib.pyplot as plt

# 读取文件并统计错误类型
file_path = '/root/Qwen3-Coder-SGLang/results/eval_prompt2_newps.jsonl'
error_count = {}
keyword_count = {}

# 定义关键词列表
keywords = ['AssertionError', 'IndentationError', 'ValueError', 'TypeError', 'SyntaxError', 'NameError', 'IndexError', 'KeyError', 'RecursionError', 'UnboundLocalError', 'ZeroDivisionError', 'Execution timed out']

with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if data['status'] == 'FAIL':
            error_type = data['error']
            if error_type in error_count:
                error_count[error_type] += 1
            else:
                error_count[error_type] = 1
            
            # 按关键词分类
            for keyword in keywords:
                if keyword in error_type:
                    if keyword in keyword_count:
                        keyword_count[keyword] += 1
                    else:
                        keyword_count[keyword] = 1

# 输出关键词分类统计结果
print("\nKeyword-based classification:")
for keyword, count in keyword_count.items():
    print(f'{keyword}: {count}')
print("\nTotal errors:")
total_errors = sum(error_count.values())
print(total_errors)

# 绘制关键词分类统计图（饼图）
# 计算总错误数
total_errors = sum(keyword_count.values())

# 分离占比大于等于2%和小于2%的数据
main_data = {k: v for k, v in keyword_count.items() if (v / total_errors) >= 0.02}
minor_data = {k: v for k, v in keyword_count.items() if (v / total_errors) < 0.02}

# 如果有占比小于2%的数据，则合并为"Others"
if minor_data:
    others_count = sum(minor_data.values())
    main_data['Others'] = others_count

# 按占比排序
sorted_data = dict(sorted(main_data.items(), key=lambda item: item[1], reverse=True))

labels = list(sorted_data.keys())
sizes = list(sorted_data.values())

# 设置颜色
colors = plt.cm.Paired(range(len(labels)))

plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, shadow=True, textprops={'fontsize': 12})

# 美化标题
plt.title('Error Classification', fontsize=16, fontweight='bold')

# 保证饼图是圆形
plt.axis('equal')  
plt.tight_layout()
plt.savefig('/root/Qwen3-Coder-SGLang/misc/error_classification.png')
plt.show()