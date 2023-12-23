import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False


def plot_dual_sentiment_fig(categories, values1, values2, save_name, title, xlabel_name='Type', ylabel_name='Value'):
    # scores_array = np.array(values)
    # min_value = np.min(scores_array, axis=0)
    # max_value = np.max(scores_array, axis=0)
    # mean_value = np.mean(scores_array, axis=0)
    # std_deviation = np.std(scores_array, axis=0)
    # normalized_scores = (scores_array - min_value) / (max_value - min_value)
    # # zscore normalization
    # z_score_normalized_scores = (scores_array - mean_value) / std_deviation

    fig, ax = plt.subplots()

    # 设置每个柱子的宽度
    bar_width = 0.35

    # 设置 x 轴位置
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

    # 绘制两组柱状图
    bars1 = ax.bar(bar_positions1, values1, width=bar_width, label='possitive sentiment')
    bars2 = ax.bar(bar_positions2, values2, width=bar_width, label='negative sentiment')

    # 添加标签和标题
    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)
    ax.set_title(title)

    # 设置 x 轴刻度标签
    ax.set_xticks(bar_positions1 + bar_width / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')

    # 添加图例
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_name)

    plt.clf()


dir_path = '/root/autodl-tmp/ProductsComment'
file_name = 'result_low.csv'

df = pd.read_csv(os.path.join(dir_path, file_name))

topics = {}
# load topic
with open(os.path.join(dir_path, file_name.split('.')[0].split('_')[-1] + 'Themes.csv')) as file:
    for line in file.readlines():
        topic = str(line.strip())
        topics[topic] = [0.0,0.0]       


phone_type_sentiment = {}
track_sentiment = {}
for row in df.itertuples():
    phone_type = row.机型
    track = row.类别
    comment = str(row.描述)
    sentiment = int(row.情感)
    score = float(row.强度)
    if phone_type in phone_type_sentiment.keys():
        phone_type_sentiment[phone_type][sentiment] += score
    else:
        phone_type_sentiment[phone_type]  = [0.0,0.0]
    if track in track_sentiment.keys():
        track_sentiment[track][sentiment] += score
    else:
        track_sentiment[track]  = [0.0, 0.0]
    for key in topics.keys():
        if comment.find(key) != -1 :
            topics[key][sentiment] += score

sorted_phone_type_sentiment = sorted(phone_type_sentiment.items(), key=lambda x: x[1][0] + x[1][0], reverse=True)
keys = [item[0] for item in sorted_phone_type_sentiment]
negative_sentiment = np.array([item[1][0] for item in sorted_phone_type_sentiment])
possitive_sentiment = np.array([item[1][1] for item in sorted_phone_type_sentiment])

plot_dual_sentiment_fig(keys, possitive_sentiment, -negative_sentiment, \
                        save_name = file_name.split('.')[0] + '_phone_type_sentiment.png',title='sentiment towards phone types', xlabel_name= 'phone type')


track_sentiment = sorted(track_sentiment.items(), key=lambda x: x[1][0] + x[1][0], reverse=True)
top_30_percent = int(0.3 * len(track_sentiment))
keys = [item[0] for item in track_sentiment[:top_30_percent]]
negative_sentiment = np.array([item[1][0] for item in track_sentiment[:top_30_percent]])
possitive_sentiment = np.array([item[1][1] for item in track_sentiment[:top_30_percent]])
plot_dual_sentiment_fig(keys, possitive_sentiment, -negative_sentiment, save_name = file_name.split('.')[0] + '_track_type_sentiment.png',\
                         title='sentiment towards tracks', xlabel_name= 'track type')

sorted_topics = sorted(topics.items(), key=lambda x: x[1][0] + x[1][0], reverse=True)
top_30_percent = int(0.1 * len(sorted_topics))
keys = [item[0] for item in sorted_topics[:top_30_percent]]
negative_sentiment = np.array([item[1][0] for item in sorted_topics[:top_30_percent]])
possitive_sentiment = np.array([item[1][1] for item in sorted_topics[:top_30_percent]])
plot_dual_sentiment_fig(keys, possitive_sentiment, -negative_sentiment, save_name = file_name.split('.')[0] + '_topics_sentiment.png',\
                         title='sentiment towards topics', xlabel_name= 'topic')

