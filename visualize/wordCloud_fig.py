import pandas as pd
import os
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image, ImageFont
import numpy as np
font = ImageFont.load_default()

dir_path = '/root/autodl-tmp/ProductsComment'
file_name = 'mid.csv'

df = pd.read_csv(os.path.join(dir_path, file_name))

# 拼接所有的描述
text = ' '.join(df['描述'].astype(str))
with open(os.path.join(dir_path,"stopWords.txt"), 'r', encoding='utf-8') as file:
    stop_words = set([line.strip() for line in file])

print(f'load {len(stop_words)} stopwords')

text = " ".join([word for word in jieba.cut(text) if word not in stop_words])

wordcloud = WordCloud(font_path="SourceHanSerifSC-VF.ttf", background_color="white", width=800, height=800, colormap="viridis",
                      mask=np.array(Image.open("background.png")), max_words=200)

# 生成词云图
wordcloud.generate(text)

# 添加标题
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("自定义词云图", fontsize=20, pad=20)

# 保存词云图到本地
wordcloud.to_file("wordcloud_" + file_name.split('.')[0] + ".png")

# 显示词云图
plt.show()
