import spacy
import jieba
from spacy.tokens import Doc
from spacy.vocab import Vocab

# 加载spaCy的中文模型
nlp = spacy.load("/root/autodl-tmp/spacy_model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0")

# 中文文本
text = "续航很不错。日常使用卡顿有一点。毕竟定位在这里，比不上旗舰机。具体看截屏，但是续航仅限于截屏中的版本。升级了新系统版本续航就相当于一个5000毫安的电池。"

# # 使用jieba进行分词
# words = list(jieba.cut(text))

# # 将jieba分词结果转换为spaCy文档
# doc = Doc(Vocab(), words=words)
# for name, proc in nlp.pipeline:
#     doc = proc(doc)
doc = nlp(text)

# 提取方面
# 这里我们关注名词及其修饰词
aspects = []
for token in doc:
    # 如果是名词，我们查找它的修饰词
    if token.pos_ == "NOUN":
        # modifiers = [child.text for child in token.children if child.dep_ in ["amod", "compound"]]
        # aspect = ''.join(modifiers + [token.text])
        # aspects.append(aspect)
        
        if token.right_edge.i - token.left_edge.i <2:
            continue
        # print(token.text)
        subtree_span = doc[token.left_edge.i : token.right_edge.i + 1]
        aspect_phrase = subtree_span.text
        aspects.append(aspect_phrase)

# 输出提取的方面
for aspect in aspects:
    print("方面:", aspect)

# 可以在此基础上加入情感分析
