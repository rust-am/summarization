import textract
from tika import parser
import re
import docx
import operator

import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import argparse

def words_count(text):
    word_list = text.split()
    number_of_words = len(word_list)
    return number_of_words

def dominant_smmarization(args, text):
    with open(args.original_text, "r", encoding="utf-8") as f:
        text = " ".join(f.readlines())

    if args.language == 'russian':
        import ru_core_news_sm
        nlp = ru_core_news_sm.load()
    else:
        import en_core_web_sm
        nlp = en_core_web_sm.load()

    doc = nlp(text)
    corpus = [sent.text.lower() for sent in doc.sents ]
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus) 
    word_list = cv.get_feature_names()

    def dominant_k(word1, word2):
        both = 0
        first = 0
        for sent in corpus:
            if ((word1 in sent) and (word2 in sent)):
                both += 1
        for sent in corpus:
            if word1 in sent:
                first += 1
        if ((both == 0) or (first == 0)):
            return(0)
        return(both/first)

    word_in_sent_count_array=[0] * len(word_list)
    for sent in corpus:
        for word in word_list:
            if word in sent:
                word_in_sent_count_array[word_list.index(word)] += 1
            else:
                continue

    word_in_sent_count = dict(zip(word_list,word_in_sent_count_array))

    words_hash = dict()
    for sentence in corpus:
        cv_fit=cv.fit_transform(corpus) 
        words = cv.get_feature_names()
        for word in words:
            if word not in words_hash:
                words_hash.update({word: {}})
            words_dup = list(words)
            words_dup.remove(word)
            list_without_word = list(words_dup)
            for word2 in list_without_word:
                if ((word in sentence) and (word2 in sentence)):
                    k = dominant_k(word, word2)
                    words_hash[word][word2] = k

    sentence_rank=[0] * len(corpus)
    for sent in corpus:
        for word in words_hash:
            for word2 in words_hash[word]:
                if ((word in sent) and (word2 in sent)):
                    sentence_rank[corpus.index(sent)] += words_hash[word][word2]

    sentences_frequency = dict(zip(corpus,sentence_rank))
    sorted_sent = dict(sorted(sentences_frequency.items(), key=operator.itemgetter(1),reverse=True))
    top_sent=sorted_sent[:args.sentences]

    summary=[]
    for sent,strength in sentences_frequency.items():  
        if strength in top_sent:
            summary.append(sent)

    return summary


def luhn_summarization(args, text):
    if args.language == 'russian':
        import ru_core_news_sm
        nlp = ru_core_news_sm.load()
    else:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    
    doc = nlp(text)
    corpus = [sent.text.lower() for sent in doc.sents ]

    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus) 
    word_list = cv.get_feature_names()
    count_list = cv_fit.toarray().sum(axis=0) 
    word_frequency = dict(zip(word_list,count_list))
    
    val=sorted(word_frequency.values())
    higher_frequency = val[-1]

    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)
    sentence_rank={}

    for sent in doc.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
            else:
                continue

    top_sentences=(sorted(sentence_rank.values())[::-1])

    top_sent=top_sentences[:args.sentences]
    
    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)

    return summary


def sumySummarize(args):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    
    parser = PlaintextParser.from_file("myfile.txt", Tokenizer(args.language))
    
    def getSummary(sumyAlgorithm):
        sumyAlgorithm.stop_words = get_stop_words(args.language)
        summary = sumyAlgorithm(parser.document, args.sentences)
        sents = " ".join([str(sentence) for sentence in summary])
        return sents

    stemmer = Stemmer(args.language)
    
    summaries = {}
    summaries['LSA'] = getSummary(LsaSummarizer(stemmer))
    summaries['TextRank'] = getSummary(TextRankSummarizer(stemmer))
    summaries['LexRank'] = getSummary(LexRankSummarizer(stemmer))
    
    words_in = "Слов: "

    print("\n===== TextRank =====")
    print(words_in + str(words_count(summaries['TextRank'])))
    print(summaries['TextRank'])
    print("")

    print("\n===== LexRank =====")
    print(words_in + str(words_count(summaries['LexRank'])))
    print(summaries['LexRank'])
    print("")

    print("\n===== LSA =====")
    print(words_in + str(words_count(summaries['LSA'])))
    print(summaries['LSA'])
    print("")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Суммаризация текста')

    parser.add_argument('--sentences', dest='sentences',
                    default=3, type=int, help='Количество предложений в аннотации (по умолчанию: 3)')

    parser.add_argument('--original_pdf', dest='original_pdf',
                    default="original.pdf", help='Путь до PDF файла')
    
    parser.add_argument('--language', dest='language',
                    default="russian", help='Язык аннотирования. Поддерживает русский или английский языки')

    parser.add_argument('--id', dest='id',
                    default="", help='Id для перехода в нужную папку')
    
    args = parser.parse_args()

    punctuation = '<>[]}{#№;-_/|()\\%ψ.Φ̃ ©ϕ×σ∂Ω±δµ∗→λ⊂α�√ε⎧⎫∫⎪ξ≤−ηΛγζΞτ£βΣθ∩*»«ßÏÄï•œГ∈=+^~—'
    raw = parser.from_file('./public/uploads/attachment/' + args.id + '/'+ args.original_pdf)
    raw_text = raw['content']
    new_raw_text = raw_text.translate(str.maketrans('', '', punctuation))
    text = new_raw_text.replace('', '')

    file = open("myfile.txt","w")
    file.writelines(text)
    file.close()

    words_in = "Слов: "
    l_summ = luhn_summarization(args, text)
    print("")
    print("\n===== Lunh modified=====")
    print(words_in + str(words_count(l_summ)))
    print(l_summ)

    d_sum = dominant_summarization(args, text)
    print("")
    print("\n===== Dominant =====")
    print(words_in + str(words_count(d_sum)))
    print(d_sum) 

    sumySummarize(args)
