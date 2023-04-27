import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.metrics import *
from sklearn.metrics import *

def read_conll_file(file_name):
    current_item = []
    with open(file_name, encoding='utf-8') as conll:
        for line in conll:
            line = line.strip()
            if line and len(line.split()) == 2: # 排除空行和參數不完整的行
                token, label = line.split()
                current_item.append((token, label))
    return current_item

def get_word_features(words):
    return dict([(word, True) for word in words])

train_set=[(get_word_features(token), label) for (token, label) in read_conll_file('e.conll')]
test_set=[(get_word_features(token), label) for (token, label) in read_conll_file('f.conll')]

classifier = nltk.NaiveBayesClassifier.train(train_set)
test_predict = []
testing = []
for item in test_set:
    test_predict.append(classifier.classify(item[0]))
    testing.append(item[1])

precision = precision_score(testing, test_predict, average=None)
recall = recall_score(testing, test_predict, average=None)
f1 = f1_score(testing, test_predict, average=None)
accuracy = nltk.classify.accuracy(classifier, test_set)

print(f'Precision： {precision}')
print(f'Recall： {recall}')
print(f'F1-score： {f1}')
print(f'Accuracy： {accuracy}')