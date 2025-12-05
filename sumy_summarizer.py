# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import argparse
import json
import time
from collections import defaultdict
import os

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.llm import LLMSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers._summarizer import AbstractSummarizer


class PromptEnhancedSummarizer(AbstractSummarizer):
    
    def __init__(self, base_summarizer, stemmer, language="english", prompt="", prompt_weight=0.5):
        super(PromptEnhancedSummarizer, self).__init__(stemmer)
        self._base_summarizer = base_summarizer
        self._language = language
        self._prompt = prompt
        self._prompt_weight = max(0.0, min(1.0, prompt_weight))
        self._prompt_keywords = self._extract_keywords(prompt)
        
    def _extract_keywords(self, prompt):
        if not prompt:
            return set()
        tokenizer = Tokenizer(self._language)
        words = tokenizer.to_words(prompt)
        stop_words = get_stop_words(self._language)
        keywords = set()
        for word in words:
            normalized = self.normalize_word(word)
            if normalized not in stop_words and len(normalized) > 2:
                keywords.add(self.stem_word(normalized))
        return keywords
    
    def _calculate_prompt_relevance(self, sentence):
        if not self._prompt_keywords:
            return 0.0
        sentence_words = set(self.stem_word(self.normalize_word(w)) 
                           for w in sentence.words)
        matches = sentence_words & self._prompt_keywords
        match_count = len(matches)
        word_freq = defaultdict(int)
        for word in sentence.words:
            stemmed = self.stem_word(self.normalize_word(word))
            if stemmed in self._prompt_keywords:
                word_freq[stemmed] += 1
        relevance_score = match_count + sum(word_freq.values()) * 0.5
        sentence_length = len(sentence.words)
        if sentence_length > 0:
            relevance_score = relevance_score / (sentence_length ** 0.5)
        return relevance_score
    
    def _get_base_ratings(self, document, sentences_count):
        base_sentences = self._base_summarizer(document, sentences_count * 2)
        ratings = {}
        for i, sentence in enumerate(base_sentences):
            ratings[sentence] = len(base_sentences) - i
        for sentence in document.sentences:
            if sentence not in ratings:
                ratings[sentence] = 0.1
        return ratings
    
    def __call__(self, document, sentences_count):
        if not self._prompt or self._prompt_weight == 0.0:
            return self._base_summarizer(document, sentences_count)
        base_ratings = self._get_base_ratings(document, sentences_count)
        prompt_ratings = {}
        max_prompt_score = 0.0
        for sentence in document.sentences:
            score = self._calculate_prompt_relevance(sentence)
            prompt_ratings[sentence] = score
            max_prompt_score = max(max_prompt_score, score)
        if max_prompt_score > 0:
            for sentence in prompt_ratings:
                prompt_ratings[sentence] = prompt_ratings[sentence] / max_prompt_score
        max_base_score = max(base_ratings.values()) if base_ratings else 1.0
        if max_base_score > 0:
            for sentence in base_ratings:
                base_ratings[sentence] = base_ratings[sentence] / max_base_score
        combined_ratings = {}
        for sentence in document.sentences:
            base_score = base_ratings.get(sentence, 0.0)
            prompt_score = prompt_ratings.get(sentence, 0.0)
            combined_ratings[sentence] = (
                (1 - self._prompt_weight) * base_score + 
                self._prompt_weight * prompt_score
            )
        return self._get_best_sentences(document.sentences, sentences_count, combined_ratings)


def build_summarizer(method, stemmer, parser, language="english", prompt="", prompt_weight=0.5):
    """Build the summarizer based on method"""
    if method == "lsa":
        base = LsaSummarizer(stemmer)
    elif method == "luhn":
        base = LuhnSummarizer(stemmer)
    elif method == "text-rank":
        base = TextRankSummarizer(stemmer)
    elif method == "lex-rank":
        base = LexRankSummarizer(stemmer)
    elif method == "sum-basic":
        base = SumBasicSummarizer(stemmer)
    elif method == "kl":
        base = KLSummarizer(stemmer)
    elif method == "edmundson":
        base = EdmundsonSummarizer(stemmer)
        stop_words = get_stop_words(language)
        base.null_words = stop_words
        if parser is not None:
            base.bonus_words = parser.significant_words
            base.stigma_words = parser.stigma_words
        else:
            base.bonus_words = ()
            base.stigma_words = ()
        return base
    elif method == "random":
        base = RandomSummarizer()
        return base
    elif method == "reduction":
        base = ReductionSummarizer(stemmer)
    elif method == "llm":
        # LLM summarizer with default LM Studio settings
        base = LLMSummarizer(
            stemmer,
            api_url="http://127.0.0.1:1234/v1/chat/completions",
            model="openai/gpt-oss-20b",
            max_tokens=500,
            temperature=0.7,
            prompt_instructions=prompt
        )
        return base
    else:
        base = LsaSummarizer(stemmer)
    
    base.stop_words = get_stop_words(language)
    
    if prompt and prompt_weight > 0:
        return PromptEnhancedSummarizer(base, stemmer, language, prompt, prompt_weight)
    return base


def generate_summary(parser, method, sentences_count, language="english", prompt="", prompt_weight=0.5):
    """Generate summary for a single configuration"""
    stemmer = Stemmer(language)
    summarizer = build_summarizer(method, stemmer, parser, language, prompt, prompt_weight)
    
    start_time = time.time()
    summary_sentences = summarizer(parser.document, sentences_count)
    elapsed_time = time.time() - start_time
    
    return {
        'method': method,
        'prompt': prompt,
        'prompt_weight': prompt_weight,
        'sentences_count': sentences_count,
        'execution_time': elapsed_time,
        'summary': [str(s) for s in summary_sentences]
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sumy Summarizer Tool")
    parser.add_argument(
        "--url",
        help="URL of the article to summarize",
    )
    parser.add_argument(
        "--article",
        default="article.txt",
        help="Path to the article file (used if --url is not provided)",
    )
    parser.add_argument(
        "--reference",
        help="Path to reference summary (optional, used for sentence count default)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Summarization methods to use (e.g. lsa lex-rank)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt for enhanced summarization",
    )
    parser.add_argument(
        "--prompt-weight",
        type=float,
        default=0.5,
        help="Weight for prompt relevance (0-1)",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=None,
        help="Number of sentences for the summary",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Language of the source text (default: english)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 100)
    print("Sumy Summarizer Tool")
    print(f"Language: {args.language}")
    print("=" * 100)
    
    # Initialize Tokenizer
    tokenizer = Tokenizer(args.language)
    
    # Determine input source
    if args.url:
        print(f"Loading content from URL: {args.url}")
        try:
            parser = HtmlParser.from_url(args.url, tokenizer)
        except Exception as e:
            print(f"Error loading URL: {e}")
            return
    else:
        print(f"Loading content from file: {args.article}")
        if not os.path.exists(args.article):
            print(f"Error: File not found: {args.article}")
            return
        parser = PlaintextParser.from_file(args.article, tokenizer)
        
    document = parser.document
    print(f"Document sentences: {len(document.sentences)}")
    
    # Determine sentence count
    sentences_count = 10  # Default default
    if args.sentences is not None:
        sentences_count = max(1, args.sentences)
        print(f"Using specified sentence count: {sentences_count}")
    elif args.reference and os.path.exists(args.reference):
        reference_parser = PlaintextParser.from_file(args.reference, tokenizer)
        sentences_count = len(reference_parser.document.sentences)
        print(f"Using sentence count from reference: {sentences_count}")
    else:
        print(f"Using default sentence count: {sentences_count}")

    # Determine methods
    default_method_names = [
        "lsa", "luhn", "edmundson", "text-rank", "lex-rank", 
        "random", "reduction", "kl", "llm"
    ]
    
    method_names = args.methods if args.methods else default_method_names
    print(f"Methods to run: {', '.join(method_names)}")
    
    if args.prompt:
        print(f"Using prompt: '{args.prompt}' (weight: {args.prompt_weight})")
        
    # Run summaries
    results = []
    print("\nGenerating summaries...")
    print("-" * 100)
    
    for method in method_names:
        print(f"Running {method}...")
        try:
            result = generate_summary(
                parser, method, sentences_count, 
                args.language, args.prompt, args.prompt_weight
            )
            results.append(result)
            print(f"  Done (Time: {result['execution_time']:.4f}s)")
        except Exception as e:
            print(f"  Error running {method}: {e}")
            
    # Save results
    output_file = "summary_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 100)
    print(f"Results saved to {output_file}")
    

if __name__ == "__main__":
    main()