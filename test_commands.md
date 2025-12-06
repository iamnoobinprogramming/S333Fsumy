### test_model_performance.py
python test_model_performance.py  --sentences=5

python test_model_performance.py --methods lsa luhn lex-rank --sentences=10

python test_model_performance.py --methods text-rank --sentences=5

python test_model_performance.py --methods text-rank --sentences=5 --prompt "focus on TextRank and LexRank" --prompt-weight 1

python test_model_performance.py --article article_chinese.txt --reference reference_chinese.txt --methods text-rank 

- `--article`: Path to the article to summarize (default: `article.txt`).
- `--reference`: Path to the reference summary (default: `reference.txt`).
- `--methods`: List of summarization methods to test (default: all available).
- `--prompt`: Optional prompt for the summarizer.
- `--prompt-weight`: Prompt weight (0.0 - 1.0, default: 0.5).
- `--sentences`: Number of sentences for the summary (default: use reference summary length).


### sumy_summarizer.py

python sumy_summarizer.py --article article.txt --sentences=5 

python sumy_summarizer.py --article article.txt --sentences=5 --prompt "focus on TextRank and LexRank"

python sumy_summarizer.py --url "https://en.wikipedia.org/wiki/Automatic_summarization" --methods text-rank --prompt "focus on TextRank and LexRank" --prompt-weight 0.7 --sentences 7 --language english

python sumy_summarizer.py --article article.txt --methods text-rank lex-rank --prompt "focus on TextRank and LexRank" --prompt-weight 0.7 --sentences 7 --language english



- `--url`: URL of the article to summarize.
- `--article`: Path to the article file (default: `article.txt`).
- `--reference`: Path to reference summary (optional, used to determine default sentence count).
- `--methods`: List of summarization methods to use (default: all available).
- `--prompt`: Text prompt to guide the summarization.
- `--prompt-weight`: Weight of the prompt (0.0 - 1.0, default: 0.5).
- `--sentences`: Number of sentences to generate.
- `--language`: Language of the source text (default: `english`).

### sumy:

### sumy_eval
sumy_eval lsa reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval luhn reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval edmundson reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval text-rank reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval lex-rank reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval random reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval reduction reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval kl reference.txt --file=article.txt --format=plaintext --language=english --length=10

sumy_eval lsa reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval luhn reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval edmundson reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10 

sumy_eval text-rank reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval lex-rank reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval random reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval reduction reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10

sumy_eval kl reference.txt --url=https://en.wikipedia.org/wiki/Automatic_summarization --language=english --length=10


### sumy
sumy lsa --length=10 --file=article.txt

sumy lex-rank --length=10 --file=article.txt

sumy luhn --length=10 --file=article.txt 

sumy edmundson --length=10 --file=article.txt 

sumy lex-rank --length=10 --file=article.txt 

sumy text-rank --length=10 --file=article.txt 

sumy random --length=10 --file=article.txt 

sumy reduction --length=10 --file=article.txt 

sumy kl --length=10 --file=article.txt 


sumy lex-rank --length=10 --url=https://en.wikipedia.org/wiki/Automatic_summarization


