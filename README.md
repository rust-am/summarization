###Суммаризация текста на основе частотности слов.

Данный скрипт на языку Python ...

# Требования 
* Spacy
* Scikit-learn

# Запуск

1. Необходимым шагом является загрузка предварительно обученных пространственных моделей общего назначения. 
Войдите в терминал и введите эту команду:  
```
./download.sh
```
Запуститься скрипт и загрузятся обученные модели.

2. Введите следующую команду, чтобы составить сводку из двух предложений:   
```
python3 summarization.py --language=russian --nb_sentences=2
```

# Документация

```
usage: summarization.py [-h] [--nb_sentences NB_SENTENCES]
                        [--original_text ORIGINAL_TEXT] [--language LANGUAGE]

Extractive text summarization

optional arguments:
  -h, --help            show this help message and exit
  --nb_sentences NB_SENTENCES
                        Number of sentences in summary (default: 3)
  --original_text ORIGINAL_TEXT
                        Path to original text
  --language LANGUAGE   Language of summary. Supports english or russian.
```
