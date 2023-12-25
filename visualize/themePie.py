import matplotlib.pyplot as plt
import sys
import os
plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False


file_path =  sys.argv[1]
themes = []
weights = []

filter = ['', "反馈", "需求", "出"]
with open(file_path, 'r') as file:
    for line in file:
        # print(line)
        theme, weight = line.strip().split(',')
        if theme in filter:
            continue
        themes.append(theme)
        weights.append(float(weight))

# 创建饼图
plt.figure(figsize=(8, 8))
max_show_num = 20
plt.pie(weights[:max_show_num], labels=themes[:max_show_num], autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # 保持饼图的纵横比相等

# 显示饼图
plt.title('主题权重分布')
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs', file_path.split(os.sep)[-1].split('.')[0]+ '_pieFig.png'))
