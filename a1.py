# Group 22
# Shayan Saberi-Nikou, SXS220123
# Mason Clark, MXC220017
# Seyed Kian Hakim, SXH200056
# Raj Thapa, RXT210036

import re
import math
from nltk import FreqDist, bigrams

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replaceWithUnk(tokens, minThreshold=1):
    """
    Replaces tokens that appear at or below minThreshold times with <UNK>.
    This builds a new list of tokens for your training set.
    """
    wordFreq = FreqDist(tokens)  # Compute frequency of each token
    # Replace low-frequency tokens with <UNK>
    processedTokens = [token if wordFreq[token] > minThreshold else '<UNK>' for token in tokens]
    return processedTokens

def smoothed_ngram_probability(ngram, unigramFreq, bigramFreq, totalWords, vocabSize, smoothing="laplace", k=0.5):
    """
    Calculate the smoothed probability for unigrams or bigrams.
    """
    if isinstance(ngram, tuple):
        # Bigram case: Calculate P(w2 | w1)
        w1, w2 = ngram
        if smoothing == "laplace":
            bigramCount = bigramFreq[ngram] + 1
            unigramCount = unigramFreq[w1] + vocabSize
        elif smoothing == "add-k":
            bigramCount = bigramFreq[ngram] + k
            unigramCount = unigramFreq[w1] + k * vocabSize
        else:
            raise ValueError("Unsupported smoothing type.")
        return bigramCount / unigramCount
    
    elif isinstance(ngram, str):
        # Unigram case: Calculate P(word)
        word = ngram
        if smoothing == "laplace":
            return (unigramFreq.get(word, 0) + 1) / (totalWords + vocabSize)
        elif smoothing == "add-k":
            return (unigramFreq.get(word, 0) + k) / (totalWords + k * vocabSize)
        else:
            raise ValueError("Unsupported smoothing type.")
    else:
        raise ValueError("ngram must be either a string (unigram) or a tuple (bigram).")

def perplexity_calculation(tokens, unigramFreq, bigramFreq, vocabSize, totalWords, n=2, smoothing="laplace", k=0.5):
    """
    Computes perplexity for either unigram or bigram model using smoothing.
    """
    log_odds = 0.0
    N = len(tokens)  # Number of tokens in the input

    for i in range(1, N):
        if n == 2:
            # Bigram case: Calculate P(tokens[i] | tokens[i-1])
            bg = (tokens[i-1], tokens[i])
            probability = smoothed_ngram_probability(bg, unigramFreq, bigramFreq, totalWords, vocabSize, smoothing, k)
        elif n == 1:
            # Unigram case: Calculate P(tokens[i])
            ug = tokens[i]
            probability = smoothed_ngram_probability(ug, unigramFreq, bigramFreq, totalWords, vocabSize, smoothing, k)
        else:
            raise ValueError("Must be a bigram (n=2) or unigram (n=1)")

        # Accumulate the log of probabilities
        log_odds += math.log(probability)

    # Perplexity formula: exp( - (1/N) * sum_of_log_probs )
    ppl = math.exp(-log_odds / N)
    return ppl

def convert_val_tokens_to_unk(val_tokens, train_vocab):
    """
    Convert tokens in validation set to <UNK> if they do not appear in the
    training vocabulary (including <UNK>).
    """
    result = []
    for t in val_tokens:
        # If token is not in training vocabulary, replace it with <UNK>
        if t not in train_vocab:
            result.append('<UNK>')
        else:
            result.append(t)
    return result

############## MAIN SCRIPT ##############

# 1) Read and preprocess train.txt
preprocessed_lines = []
with open('train.txt', 'r') as file:
    for line in file:
        cleaned_line = preprocess_text(line)  # Clean and normalize text
        preprocessed_lines.append(cleaned_line)

# Flatten into a list of tokens
train_tokens = [token for line in preprocessed_lines for token in line.split()]

# 2) Replace low-frequency tokens with <UNK> in the training set
processedTokens = replaceWithUnk(train_tokens, minThreshold=3)

# 3) Build frequency distributions for unigrams and bigrams
unigramFreq = FreqDist(processedTokens)  # Frequency of individual words
bigramFreq = FreqDist(bigrams(processedTokens))  # Frequency of consecutive word pairs

# 4) Calculate total words and vocabulary size
totalWords = len(processedTokens)  # Total number of tokens in the processed training set
vocabSize = len(unigramFreq)  # Number of unique words (vocabulary size)

print("\nTotal Words in Training:", totalWords)
print("Vocabulary Size (training):", vocabSize)

# Example: Print out some probabilities from the training model
print("\nSmoothed Unigram Probabilities (Laplace, sample):")
some_words = list(unigramFreq.keys())[:10]  # Select a subset of words for demonstration
for w in some_words:
    probUni = smoothed_ngram_probability(w, unigramFreq, bigramFreq, totalWords, vocabSize, smoothing="laplace")
    print(f"P({w}) = {probUni:.4f}")

print("\nSmoothed Bigram Probabilities (Add-k, sample, k=0.5):")
some_bigrams = list(bigramFreq.keys())[:10]  # Select a subset of bigrams for demonstration
for bg in some_bigrams:
    probBi = smoothed_ngram_probability(bg, unigramFreq, bigramFreq, totalWords, vocabSize, smoothing="add-k", k=0.5)
    print(f"P({bg[1]} | {bg[0]}) = {probBi:.4f}")

# 5) Now do the same for the validation set
val_lines = []
with open('val.txt', 'r') as f:
    for line in f:
        val_lines.append(preprocess_text(line))  # Preprocess validation data

val_tokens_raw = [token for line in val_lines for token in line.split()]

# Replace tokens not in the training vocab with <UNK>
train_vocab = set(unigramFreq.keys())  # Training vocabulary including <UNK>
valTokens_unk = convert_val_tokens_to_unk(val_tokens_raw, train_vocab)

print("\nNumber of tokens in Validation (raw):", len(val_tokens_raw))
print("Number of tokens in Validation (UNKed):", len(valTokens_unk))

# 6) Compute perplexities on the UNKified validation data
unigram_pp_laplace = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=1, smoothing="laplace")
bigram_pp_laplace  = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=2, smoothing="laplace")

unigram_pp_addk_05 = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=1, smoothing="add-k", k=0.5)
bigram_pp_addk_05  = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=2, smoothing="add-k", k=0.5)

unigram_pp_addk_01 = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=1, smoothing="add-k", k=0.1)
bigram_pp_addk_01  = perplexity_calculation(valTokens_unk, unigramFreq, bigramFreq, vocabSize, totalWords, n=2, smoothing="add-k", k=0.1)

# Display perplexity results
print(f"\nUnigram Perplexity on validation set (Laplace): {unigram_pp_laplace:.4f}")
print(f"Bigram Perplexity on validation set (Laplace): {bigram_pp_laplace:.4f}")

print(f"\nUnigram Perplexity on validation set (Add-k, k=0.5): {unigram_pp_addk_05:.4f}")
print(f"Bigram Perplexity on validation set (Add-k, k=0.5): {bigram_pp_addk_05:.4f}")

print(f"\nUnigram Perplexity on validation set (Add-k, k=0.1): {unigram_pp_addk_01:.4f}")
print(f"Bigram Perplexity on validation set (Add-k, k=0.1): {bigram_pp_addk_01:.4f}")

# Report and compare results
print("\n### Perplexity Comparison ###")
print(f"Laplace Unigram Perplexity: {unigram_pp_laplace:.4f}")
print(f"Laplace Bigram Perplexity: {bigram_pp_laplace:.4f}")
print(f"Add-k Unigram Perplexity (k=0.5): {unigram_pp_addk_05:.4f}")
print(f"Add-k Bigram Perplexity (k=0.5): {bigram_pp_addk_05:.4f}")
print(f"Add-k Unigram Perplexity (k=0.1): {unigram_pp_addk_01:.4f}")
print(f"Add-k Bigram Perplexity (k=0.1): {bigram_pp_addk_01:.4f}")
