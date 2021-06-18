import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import argparse


def summarization(args):
    
    with open(args.original_text, "r", encoding="utf-8") as f:
        text = " ".join(f.readlines())

    if args.language == 'russian':
        import ru_core_news_sm
        nlp = ru_core_news_sm.load()
    else:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    
    
    doc = nlp(text)
    
    # lowercase
    corpus = [sent.text.lower() for sent in doc.sents ]
    # 
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   

    # 
    cv_fit=cv.fit_transform(corpus) 

    # 
    word_list = cv.get_feature_names()

    # 
    count_list = cv_fit.toarray().sum(axis=0)    

    """
    Функция zip (* iterables) принимает итераторы в качестве аргументов и возвращает др итератор.
    Этот итератор генерирует серию хэшей(асоц. массив), содержащих элементы из каждой итерации.
    Преобразуем эти кортежи в словарь {word: frequency} 
    """

    word_frequency = dict(zip(word_list,count_list))
    
    val=sorted(word_frequency.values())


    # проверяет слова с более высокой частотой 
    higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
    print("\nСлова с более высокой частотой: ", higher_word_frequencies)

    # получает относительные частоты слов 
    higher_frequency = val[-1]
    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)


    # SENTENCE RANKING:ранжирование предложений, основано на частотности слов
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

    # собираем абстракт
    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)

    print(summary)
    # возвращаем оригинальный текст и абстракт
    return text, summary


if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description='Extractive text summarization')

    parser.add_argument('--sentences', dest='sentences',
                    default=3, type=int, help='Number of sentences in summary (default: 3)')

    parser.add_argument('--original_text', dest='original_text',
                    default="original_text.txt", help='Path to original text')
    
    parser.add_argument('--language', dest='language',
                    default="russian", help='Language of summary. Supports russian or english')

    parser.add_argument('--heavy_words ', dest='heavy_words',
                    default=3, type=int, help='Words dictionary with given weights: {\'word\' = 0.3, \'something\' = 0.7}')
    
    args = parser.parse_args()

    text, summary = summarization(args)

    # вывод
    print("========= Original text ========")
    print(text)

    print("\n============ Summary ============")
  
    for i in summary:
        print(i,end=" ")
    print("\n========= End of summary =========")
