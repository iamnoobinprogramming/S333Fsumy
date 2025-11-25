# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import json
try:
    import requests
except ImportError:
    requests = None

from ._summarizer import AbstractSummarizer
from ..nlp.stemmers import null_stemmer
from .._compat import to_unicode


class LLMSummarizer(AbstractSummarizer):
    """
    Summarizer using local LLM via LM Studio or OpenAI-compatible API.
    
    This summarizer sends the document to a local LLM server and uses
    the generated abstractive summary. The summary is then converted
    back to extractive format by matching sentences from the original document.
    """
    
    def __init__(self, stemmer=null_stemmer, api_url="http://127.0.0.1:1234/v1/chat/completions", 
                 model="openai/gpt-oss-20b", max_tokens=500, temperature=0.7,
                 max_context_tokens=28000, use_chunking=True, prompt_instructions=""):
        """
        Args:
            stemmer: Stemmer instance (not used but required by AbstractSummarizer)
            api_url: URL of the local LLM API endpoint
            model: Model identifier for the API
            max_tokens: Maximum tokens for the summary response
            temperature: Sampling temperature (0.0-1.0)
            max_context_tokens: Maximum tokens to use for input (default 28000, leaving room for 32768 total)
            use_chunking: If True, split long documents into chunks and summarize each (default: True)
            prompt_instructions: Optional extra instructions appended to every prompt
        """
        super(LLMSummarizer, self).__init__(stemmer)
        self._api_url = api_url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_context_tokens = max_context_tokens
        self._use_chunking = use_chunking
        self._prompt_instructions = prompt_instructions.strip()
        self._stop_words = frozenset()
    
    @property
    def stop_words(self):
        return self._stop_words
    
    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))
    
    def _ensure_dependencies_installed(self):
        """Check if requests library is installed"""
        if requests is None:
            raise ValueError(
                "LLM summarizer requires 'requests' library. "
                "Please install it by: pip install requests"
            )
    
    def _call_llm_api(self, prompt):
        """
        Call the local LLM API to generate summary.
        
        Args:
            prompt: The prompt text to send to the LLM
            
        Returns:
            Generated text from the LLM
        """
        self._ensure_dependencies_installed()
        
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self._api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text from the response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Unexpected API response format")
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to call LLM API: {e}")
    
    def _estimate_tokens(self, text):
        """
        Rough estimation of token count (approximately 4 characters per token).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _split_document_into_chunks(self, document, chunk_size_sentences=80):
        """
        Split long document into overlapping chunks for processing.
        Each chunk will be summarized separately, then combined.
        
        Args:
            document: Document to split
            chunk_size_sentences: Number of sentences per chunk
            
        Returns:
            List of document chunks (each as a list of sentences)
        """
        sentences = list(document.sentences)
        if len(sentences) <= chunk_size_sentences:
            return [sentences]
        
        chunks = []
        overlap = chunk_size_sentences // 4  # 25% overlap between chunks
        
        for i in range(0, len(sentences), chunk_size_sentences - overlap):
            chunk = sentences[i:i + chunk_size_sentences]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _truncate_document(self, document, max_sentences=100):
        """
        Truncate document if it's too long to fit in context window.
        Uses first N sentences or extracts key sentences using simple heuristics.
        
        Args:
            document: Document to truncate
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Truncated text string
        """
        if len(document.sentences) <= max_sentences:
            # Document is short enough, use all sentences
            return " ".join(str(sentence) for sentence in document.sentences)
        
        # Document is too long, use first and last sentences + some middle ones
        # This ensures we capture introduction and conclusion
        sentences = document.sentences
        first_part = sentences[:max_sentences // 2]
        last_part = sentences[-max_sentences // 2:]
        
        truncated = list(first_part) + list(last_part)
        return " ".join(str(sentence) for sentence in truncated)
    
    def _build_instruction_text(self, sentences_count):
        base_instruction = (
            f"Please summarize the following text in approximately {sentences_count} sentences. "
            "if you are given a prompt, follow the prompt. Otherwise, focus on the most important information and key points."
        )
        if self._prompt_instructions:
            return (
                f"{base_instruction}\nAdditional requirements: {self._prompt_instructions}"
            )
        return base_instruction
    
    def _create_summary_prompt(self, document, sentences_count, max_context_tokens=28000):
        """
        Create a prompt for the LLM to generate a summary.
        Automatically truncates long documents to fit within context window.
        
        Args:
            document: The document to summarize
            sentences_count: Target number of sentences in summary
            max_context_tokens: Maximum tokens to use for input (leaving room for prompt and response)
            
        Returns:
            Prompt string
        """
        # Combine all sentences from the document
        full_text = " ".join(str(sentence) for sentence in document.sentences)
        
        # Estimate tokens and truncate if necessary
        estimated_tokens = self._estimate_tokens(full_text)
        
        if estimated_tokens > max_context_tokens:
            # Document is too long, truncate it
            # Use approximately max_sentences based on average sentence length
            avg_chars_per_sentence = len(full_text) / len(document.sentences) if document.sentences else 0
            max_sentences = int((max_context_tokens * 4) / avg_chars_per_sentence) if avg_chars_per_sentence > 0 else 100
            max_sentences = max(50, min(max_sentences, len(document.sentences)))  # At least 50, at most all sentences
            
            truncated_text = self._truncate_document(document, max_sentences)
        else:
            truncated_text = full_text
        
        instruction = self._build_instruction_text(sentences_count)
        
        prompt = f"""{instruction}

Text:
{truncated_text}

Summary:"""
        
        return prompt
    
    def _match_summary_to_sentences(self, summary_text, document, tokenizer):
        """
        Match the LLM-generated summary to original document sentences.
        This converts abstractive summary back to extractive format.
        
        Args:
            summary_text: The generated summary text from LLM
            document: Original document
            tokenizer: Tokenizer instance
            
        Returns:
            Tuple of Sentence objects that best match the summary
        """
        # Split summary into sentences
        summary_sentences = tokenizer.to_sentences(summary_text)
        
        matched_sentences = []
        for summary_sent in summary_sentences:
            # Find the most similar sentence from the original document
            best_match = None
            best_score = 0.0
            
            summary_words = set(tokenizer.to_words(summary_sent.lower()))
            
            for orig_sent in document.sentences:
                orig_words = set(tokenizer.to_words(str(orig_sent).lower()))
                
                # Calculate simple word overlap score
                if len(summary_words) > 0:
                    overlap = len(summary_words & orig_words) / len(summary_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_match = orig_sent
            
            # Only add if similarity is above threshold
            if best_match and best_score > 0.3:
                matched_sentences.append(best_match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sent in matched_sentences:
            if sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)
        
        return tuple(unique_sentences)
    
    def __call__(self, document, sentences_count):
        """
        Generate summary using local LLM.
        Uses chunking for long documents to process the entire text.
        
        Args:
            document: Document to summarize
            sentences_count: Target number of sentences
            
        Returns:
            Tuple of Sentence objects
        """
        if len(document.sentences) == 0:
            return ()
        
        tokenizer = document.sentences[0]._tokenizer
        
        # Check if document is too long
        full_text = " ".join(str(sentence) for sentence in document.sentences)
        estimated_tokens = self._estimate_tokens(full_text)
        
        if self._use_chunking and estimated_tokens > self._max_context_tokens:
            # Use chunking approach: split document, summarize each chunk, then combine
            chunks = self._split_document_into_chunks(document)
            all_summary_texts = []
            
            # Calculate sentences per chunk (distribute target across chunks)
            sentences_per_chunk = max(2, sentences_count // len(chunks))
            
            for i, chunk in enumerate(chunks):
                # Create a temporary document from this chunk
                chunk_text = " ".join(str(s) for s in chunk)
                
                # Create prompt for this chunk
                instruction = self._build_instruction_text(sentences_per_chunk)
                chunk_prompt = f"""{instruction}

Text:
{chunk_text}

Summary:"""
                
                # Call LLM for this chunk
                try:
                    chunk_summary = self._call_llm_api(chunk_prompt)
                    all_summary_texts.append(chunk_summary)
                except Exception as e:
                    # If chunk fails, skip it
                    continue
            
            # Combine all chunk summaries
            combined_summary = " ".join(all_summary_texts)
        else:
            # Document is short enough or chunking disabled, use single call
            prompt = self._create_summary_prompt(document, sentences_count, self._max_context_tokens)
            combined_summary = self._call_llm_api(prompt)
        
        # Convert abstractive summary to extractive format
        # by matching to original sentences
        matched_sentences = self._match_summary_to_sentences(
            combined_summary, document, tokenizer
        )
        
        # If we got fewer sentences than requested, try to add more
        if len(matched_sentences) < sentences_count:
            # Add remaining sentences from document in order
            remaining = sentences_count - len(matched_sentences)
            for sent in document.sentences:
                if sent not in matched_sentences and remaining > 0:
                    matched_sentences = matched_sentences + (sent,)
                    remaining -= 1
        
        # Limit to requested count
        return matched_sentences[:sentences_count]

