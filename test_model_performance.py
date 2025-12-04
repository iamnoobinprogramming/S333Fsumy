from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import argparse
import json
import time
from collections import defaultdict
from itertools import chain

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
from sumy.models import TfDocumentModel

# Import evaluation metrics
from sumy.evaluation import (
    rouge_1, rouge_2, rouge_l_sentence_level, rouge_l_summary_level,
    precision, recall, f_score, cosine_similarity, unit_overlap
)


LANGUAGE = "english"


class PromptEnhancedSummarizer(AbstractSummarizer):
    
    def __init__(self, base_summarizer, stemmer, prompt="", prompt_weight=0.5):
        super(PromptEnhancedSummarizer, self).__init__(stemmer)
        self._base_summarizer = base_summarizer
        self._prompt = prompt
        self._prompt_weight = max(0.0, min(1.0, prompt_weight))
        self._prompt_keywords = self._extract_keywords(prompt)
        
    def _extract_keywords(self, prompt):
        if not prompt:
            return set()
        tokenizer = Tokenizer(LANGUAGE)
        words = tokenizer.to_words(prompt)
        stop_words = get_stop_words(LANGUAGE)
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


def build_summarizer(method, stemmer, parser, prompt="", prompt_weight=0.5):
    """Build summarizer"""
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
        stop_words = get_stop_words(LANGUAGE)
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
    
    base.stop_words = get_stop_words(LANGUAGE)
    
    if prompt and prompt_weight > 0:
        return PromptEnhancedSummarizer(base, stemmer, prompt, prompt_weight)
    return base


def evaluate_summary(evaluated_sentences, reference_sentences, document_sentences=None):
    """Evaluate summary quality, return multiple metrics"""
    results = {}
    
    try:
        results['rouge_1'] = rouge_1(evaluated_sentences, reference_sentences)
    except:
        results['rouge_1'] = 0.0
    
    try:
        results['rouge_2'] = rouge_2(evaluated_sentences, reference_sentences)
    except:
        results['rouge_2'] = 0.0
    
    try:
        results['rouge_l_sentence'] = rouge_l_sentence_level(evaluated_sentences, reference_sentences)
    except:
        results['rouge_l_sentence'] = 0.0
    
    try:
        results['rouge_l_summary'] = rouge_l_summary_level(evaluated_sentences, reference_sentences)
    except:
        results['rouge_l_summary'] = 0.0
    
    try:
        results['precision'] = precision(evaluated_sentences, reference_sentences)
    except:
        results['precision'] = 0.0
    
    try:
        results['recall'] = recall(evaluated_sentences, reference_sentences)
    except:
        results['recall'] = 0.0
    
    try:
        results['f_score'] = f_score(evaluated_sentences, reference_sentences)
    except:
        results['f_score'] = 0.0
    
    # Document-based evaluation (requires full document)
    if document_sentences:
        try:
            evaluated_words = tuple(chain(*(s.words for s in evaluated_sentences)))
            document_words = tuple(chain(*(s.words for s in document_sentences)))
            evaluated_model = TfDocumentModel(evaluated_words)
            document_model = TfDocumentModel(document_words)
            results['cosine_similarity'] = cosine_similarity(evaluated_model, document_model)
        except:
            results['cosine_similarity'] = 0.0
        
        try:
            results['unit_overlap'] = unit_overlap(evaluated_model, document_model)
        except:
            results['unit_overlap'] = 0.0
    
    return results


def test_configuration(parser, document, reference_sentences, method, sentences_count, 
                      prompt="", prompt_weight=0.5):
    """Test single configuration"""
    stemmer = Stemmer(LANGUAGE)
    summarizer = build_summarizer(method, stemmer, parser, prompt, prompt_weight)
    
    # Timing
    start_time = time.time()
    evaluated_sentences = summarizer(document, sentences_count)
    elapsed_time = time.time() - start_time
    
    # Evaluation
    metrics = evaluate_summary(evaluated_sentences, reference_sentences, document.sentences)
    metrics['execution_time'] = elapsed_time
    metrics['summary_length'] = len(evaluated_sentences)
    
    return {
        'method': method,
        'prompt': prompt,
        'prompt_weight': prompt_weight,
        'sentences_count': sentences_count,
        'metrics': metrics,
        'summary': [str(s) for s in evaluated_sentences]
    }


def print_results_table(results):
    """Print results table"""
    print("\n" + "=" * 100)
    print("Evaluation Summary")
    print("=" * 100)
    
    # Header
    header = f"{'Method':<15} {'Prompt':<20} {'Weight':<8} {'ROUGE-1':<10} {'ROUGE-2':<10} {'F-Score':<10} {'Time(s)':<10}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for result in results:
        method = result['method']
        prompt = result['prompt'][:18] if result['prompt'] else "None"
        weight = result['prompt_weight']
        m = result['metrics']
        print(f"{method:<15} {prompt:<20} {weight:<8.2f} {m['rouge_1']:<10.4f} "
              f"{m['rouge_2']:<10.4f} {m['f_score']:<10.4f} {m['execution_time']:<10.4f}")
    
    print("=" * 100)



def save_results_to_json(results, filename="evaluation_results.json"):
    """Save results to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sumy Summarizer Performance Test")
    parser.add_argument(
        "--article",
        default="article.txt",
        help="Path to the article to summarize (default: article.txt)",
    )
    parser.add_argument(
        "--reference",
        default="reference.txt",
        help="Path to the reference summary (default: reference.txt)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Summarization methods to test (empty for all). E.g.: --methods lex-rank",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt for the summarizer",
    )
    parser.add_argument(
        "--prompt-weight",
        type=float,
        default=0.5,
        help="Prompt weight (0-1), only effective when --prompt is provided",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=None,
        help="Number of sentences for the summary (default: use reference summary length)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 100)
    print("Sumy Summarization Performance Test")
    print("=" * 100)
    
    # ========================================================================
    # Configure test parameters
    # ========================================================================
    
    print("\nLoading article and reference from local files...")
    tokenizer = Tokenizer(LANGUAGE)
    parser = PlaintextParser.from_file(args.article, tokenizer)
    document = parser.document
    
    reference_parser = PlaintextParser.from_file(args.reference, tokenizer)
    reference_sentences = reference_parser.document.sentences
    
    if not reference_sentences:
        raise ValueError("Reference summary is empty. Please check the reference file.")
    
    # Use user-specified sentence count, or default to reference summary length
    if args.sentences is not None:
        sentences_count = max(1, args.sentences)  # At least 1 sentence
        print(f"\nUsing custom sentence count: {sentences_count}")
    else:
        sentences_count = len(reference_sentences)
    
    print(f"Document sentences: {len(document.sentences)}")
    print(f"Reference sentences: {len(reference_sentences)}")
    print(f"Each summary will output {sentences_count} sentences for fair comparison.")
    
    # Summarization methods to test (all)
    default_method_names = [
        "lsa",
        "luhn",
        "edmundson",
        "text-rank",
        "lex-rank",
        "random",
        "reduction",
        "kl",
        "llm",  # Local LLM method (requires LM Studio running)
    ]
    
    if args.methods:
        method_names = args.methods
        print(f"\nUsing custom methods: {', '.join(method_names)}")
    else:
        method_names = default_method_names
    
    # Test configuration list (unified without prompt)
    prompt_text = args.prompt.strip()
    prompt_weight = max(0.0, min(1.0, args.prompt_weight)) if prompt_text else 0.0
    if prompt_text:
        print(f"\nUsing prompt (weight={prompt_weight:.2f}): {prompt_text}")
    test_configs = [
        (method, prompt_text, prompt_weight, sentences_count)
        for method in method_names
    ]
    
    # ========================================================================
    # Run tests
    # ========================================================================
    
    results = []
    print("\nRunning evaluations...")
    print("-" * 100)
    
    for i, (method, prompt, weight, count) in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] Method={method}, prompt='{prompt}', weight={weight}")
        try:
            result = test_configuration(
                parser, document, reference_sentences, method, count, prompt, weight
            )
            results.append(result)
            print(f"  Done (ROUGE-1: {result['metrics']['rouge_1']:.4f})")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # ========================================================================
    # Show results
    # ========================================================================
    
    if results:
        print_results_table(results)
        
        # Save results
        save_results_to_json(results)
    else:
        print("\nNo successful evaluation results.")


if __name__ == "__main__":
    main()


