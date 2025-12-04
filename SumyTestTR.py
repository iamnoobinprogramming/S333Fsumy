from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

#from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank_mod2 import TextRankSummarizermod as Summarizermod #Modified
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizerori #Original 
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
nltk.download('punkt_tab')

#Evaluation
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
from sumy.evaluation.rouge import rouge_1, rouge_2
from sumy.evaluation.coselection import precision, recall, f_score
from sumy.evaluation.content_based import cosine_similarity, unit_overlap
from sumy.models import TfDocumentModel

chinesetxt = '''
很久很久以前，三隻小猪和牠们的妈妈一
起住在家裡。有一天，妈妈说：「你们已经长
大了，不能再住在家裡。你们要出去找自己的
房子。」她亲吻每隻小猪和牠们道别，并提醒
牠们要远离大野狼。 
三隻小猪在路上走着走着，牠们看到一个
男人驾着装满稻草的四轮车。第一隻小猪对这
个人说：「请你将你的稻草卖给我。」于是那
个人将稻草卖给了第一隻小猪。稻草虽然无法
建造一个坚固的房子，但是却很便宜。于是第
一隻小猪用稻草盖了一间房子，并将剩下的钱
拿去买糖果零食。 
第二隻小猪看到一个男人驾着装满树枝的
四轮车。第二隻小猪对这个人说：「请你将你
的树枝卖给我。」于是那个人将树枝卖给了第
二隻小猪。树枝虽然无法建造一个坚固的房子
，但是却很便宜。于是第二隻小猪用树枝盖了
一间房子，并将剩下的钱拿去买速食和饮料。 
第三隻小猪看到一个男人驾着装满砖块的
四轮车。第三隻小猪对这个人说：「请你将你
的砖块卖给我。」于是那个人将砖块卖给了第
三隻小猪。砖块虽然可以建造一个坚固的房子
，但是却很贵。于是第三隻小猪将所有的钱花
在砖块上，并用砖块盖了一间房子。第三隻小
猪因为没钱必须到野外觅食。牠找到野生的茎
、葱和香菰。牠吃得非常的健康。 
突然，大野狼来到了第一隻小猪的门口。
牠敲敲门说：「小猪啊小猪，让我进去吧！让
我进去吧！」 
第一隻小猪回答：「不要！不管你怎麽说
，我绝对不会开门。」
大野狼说：「小猪啊，如果你不让我进去
，我会一直吹，把你的房子都吹倒。」然后大
野狼一直吹，把房子都吹倒了。第一隻小猪很
快的跑到牠弟弟第二隻小猪的家。 
门一关上，大野狼就出现了。牠敲敲门说
：「小猪啊小猪，让我进去吧！让我进去吧！
」 
两隻小猪同时回答：「不要！不管你怎麽
说，我们绝对不会开门。」 
大野狼说：「小猪们啊，如果你不让我进
去，我会一直吹，把你的房子都吹倒。」然后
大野狼一直吹，把房子都吹倒了。两隻小猪很
快的跑到牠们弟弟第三隻小猪的家。 
第三隻小猪的家门一关上，大野狼就出现
了。牠敲敲门说：「小猪们啊小猪，让我进去
吧！让我进去吧！」 
三隻小猪同时回答：「不要！不管你怎麽
说，我们绝对不会开门。」 
野狼说：「小猪们啊，如果你们不让我进
去，我会一直吹，把你们的房子都吹倒。」然
后大野狼一直吹，一直吹一直吹，但是不管牠
怎麽吹就是无法把房子吹倒。于是大野狼假装
很友善的说：「小猪们啊，我知道哪裡有很棒
的萝蔔园喔。」 
「哪裡？」三隻小猪问。 
「在山丘的另一边。我明天会再过来，我
们可以一起去那边。」 
「很好。」三隻小猪说：「我们会准备好，
你几点会来？」
大野狼回答六点钟，但是三隻小猪五点就
起床，在野狼来之前将萝蔔拔起。 
大野狼说：「小猪们啊，我要和你们一起
去拔萝蔔。」 
三隻小猪说：「我们已经去拔完萝蔔回来
了，并将萝蔔煮成晚餐。」 
大野狼非常生气。牠想要吃掉这三隻小猪
。牠从房子的旁边爬到屋顶。当牠正要爬下烟
囱的时候，三隻小猪听到牠的声音。于是他们
将装有滚烫热水的锅子挂在烟囱的下面。正当
大野狼要下来的时候，三隻小猪将锅盖打开。
大野狼掉进了热水裡，牠大声的惨叫爬回烟囱
上。 
整条路上都可以听到大野狼在惨叫，直到
今天还是可以听到大野狼的惨叫声。三隻小猪
从此快乐的生活着。 
'''

referencetxt = '''很久很久以前，三隻小猪和牠们的妈妈一
起住在家里。她亲吻每隻小猪和牠们道别，并提醒
牠们要远离大野狼。于是第
一隻小猪用稻草盖了一间房子，并将剩下的钱
拿去买糖果零食。于是第二隻小猪用树枝盖了
一间房子，并将剩下的钱拿去买速食和饮料。
于是第三隻小猪将所有的钱花
在砖块上，并用砖块盖了一间房子。
突然，大野狼来到了第一隻小猪的门口。
然后大
野狼一直吹，把房子都吹倒了。第一隻小猪很
快的跑到牠弟弟第二隻小猪的家。
然后
大野狼一直吹，把房子都吹倒了。两隻小猪很
快的跑到牠们弟弟第三隻小猪的家。
第三隻小猪的家门一关上，大野狼就出现
了。
」然
后大野狼一直吹，一直吹一直吹，但是不管牠
怎麽吹就是无法把房子吹倒。于是大野狼假装
很友善的说：「小猪们啊，我知道哪里有很棒
的萝蔔园喔。」
大野狼回答六点钟，但是三隻小猪五点就
起床，在野狼来之前将萝蔔拔起。
三隻小猪说：「我们已经去拔完萝蔔回来
了，并将萝蔔煮成晚餐。」
牠想要吃掉这三隻小猪。
正当大野狼要下来的时候，三隻小猪将锅盖打开。
大野狼掉进了热水里，牠大声的惨叫爬回烟囱
上。
整条路上都可以听到大野狼在惨叫，直到
今天还是可以听到大野狼的惨叫声。三隻小猪
从此快乐的生活着。 
'''
LANGUAGE = "chinese"
SENTENCES_COUNT = 20


if __name__ == "__main__":

    stemmer = Stemmer(LANGUAGE)

    #Parser
    doc = PlaintextParser.from_string(chinesetxt, Tokenizer(LANGUAGE))#Text
    ref = PlaintextParser.from_string(referencetxt, Tokenizer(LANGUAGE))#Text
    #docart = PlaintextParser.from_file("oriarticle.txt", Tokenizer(LANGUAGE))#File
    #refart = PlaintextParser.from_file("20sentences.txt", Tokenizer(LANGUAGE))#File
    
    #chinese summarize
    summarizer1 = Summarizermod(stemmer, alpha = 2, rate=0.68)#After modify textrank, default alpha = 1, rate = 0
    #summarizer1 = Summarizermod(stemmer, alpha = 1, rate=0)
    summarizer = Summarizerori(stemmer)#Before modify textrank
    summarizer1.stop_words = get_stop_words(LANGUAGE)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    #Evaluate
    sumtxt1 = summarizer1(doc.document, SENTENCES_COUNT)
    sumtxt = summarizer(doc.document, SENTENCES_COUNT)
    #sumtxt1 = summarizer1(docart.document, SENTENCES_COUNT)
    #sumtxt = summarizer(docart.document, SENTENCES_COUNT)

    def cosine_sim(evaldoc, refdoc):
        evaled = tuple(chain(*(s.words for s in evaldoc)))
        ref = tuple(chain(*(s.words for s in refdoc)))
        evalmodel = TfDocumentModel(evaled)
        refmodel = TfDocumentModel(ref)
        return cosine_similarity(evalmodel, refmodel)
        
    def unit_Overlap(evaldoc, refdoc):
        evaled = tuple(chain(*(s.words for s in evaldoc)))
        ref = tuple(chain(*(s.words for s in refdoc)))
        evalmodel = TfDocumentModel(evaled)
        refmodel = TfDocumentModel(ref)
        return unit_overlap(evalmodel, refmodel)

    def eval_summarizer(sumtxt, ref, sentence_count=20):

        scores = {
                'ROUGE-1': rouge_1(sumtxt, ref.document.sentences),
                'ROUGE-2': rouge_2(sumtxt, ref.document.sentences),
                'Precision': precision(sumtxt, ref.document.sentences),
                'Recall': recall(sumtxt, ref.document.sentences),
                'F-Score': f_score(sumtxt, ref.document.sentences),
                'Cosine Similarity': cosine_sim(sumtxt, ref.document.sentences),
                'Unit Overlap': unit_Overlap(sumtxt, ref.document.sentences)
        }
        return scores
        
    def draw_graph(result1, result2, label1='Result1', label2='Result2', title='Title'):

        metrics = ['ROUGE-1', 'ROUGE-2', 'Precision', 'Recall', 'F-Score', 'Cosine Similarity', 'Unit Overlap']

        value1 = [result1[metric] for metric in metrics]
        value2 = [result2[metric] for metric in metrics]

        x = np.arange(len(metrics))
        width=0.5

        fig, ax = plt.subplots(figsize=(14, 7))

        bar1 = ax.bar(x-width/2, value1, width, label=label1, edgecolor='black', color='#b3f0ff')
        bar2 = ax.bar(x+width/3, value2, width, label=label2, edgecolor='black', color='#3366cc')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.55, edgecolor='black', fancybox=True)


    result1 = eval_summarizer(sumtxt, ref, 20)
    result2 = eval_summarizer(sumtxt1, ref, 20)
    
    plot = draw_graph(result1, result2, label1='Original Textrank', label2='Modified Textrank', title='Comparing Original Textrank and Modified Textrank')
    plt.savefig('result.png')
    plt.show()
    
    
    print(f'Original Textrank: {result1}')
    print('\n')
    print(f'Modified Textrank: {result2}')
    print('\n')
    
    #Result of the summary
    print(f'Original Textrank summary:')
    for index, sentence in enumerate(sumtxt):
        print(f'{index+1} {sentence}')
    print('\n')
    print(f'Modified Textrank summary:')
    for index, sentence in enumerate(sumtxt1):
        print(f'{index+1} {sentence}')
    print('\n')
    print(f'Reference text:')
    for index, sentence in enumerate(ref.document.sentences):
        print(f'{index+1} {sentence}')
