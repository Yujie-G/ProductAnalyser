from gensim import corpora, models
import jieba
import pandas as pd
import os

# path to the data file
dir_path = '/root/autodl-tmp/ProductsComment'

file_name = 'high.csv'
save_name = 'highThemes.csv'
# 读取CSV文件
df = pd.read_csv(os.path.join(dir_path, file_name))

# 拼接所有的描述
text = ' '.join(df['描述'].astype(str))


with open(os.path.join(dir_path,"stopWords.txt"), 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file])

# 分词
words = [word for word in jieba.cut(text) if word not in stopwords]

# 创建文档词汇表
dictionary = corpora.Dictionary([words])

# 创建文档-词汇矩阵
corpus = [dictionary.doc2bow(words)]

# 训练LDA模型
lda = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)

# 提取主题
topics = lda.print_topics(num_words=200)

with open(os.path.join(dir_path, save_name), "w", encoding='utf-8') as file:
    for topic in topics:
        sent = topic[1].split(' + ')
        for a_mult_b in sent:
            value = a_mult_b.split('*')[0]
            theme = a_mult_b.split('*')[-1].strip('"')
            file.write(theme +','+ value + '\n')
            
