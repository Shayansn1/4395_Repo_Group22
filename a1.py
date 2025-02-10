# Group 22
# Shayan Saberi-Nikou, SXS220123
# Mason Clark, MXC220017
# Seyed Kian Hakim, SXH200056
# Raj Thapa, RXT210036

import re
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
    wordFreq = FreqDist(tokens)
    processedTokens = [token if wordFreq[token] > minThreshold else '<UNK>' for token in tokens]
    return processedTokens

def laplaceSmoothingUnigram(word, unigramFreq, totalWords, vocabSize):
    return (unigramFreq.get(word, 0) + 1) / (totalWords + vocabSize)

def laplaceSmoothingBigram(bigram, bigramFreq, unigramFreq, vocabSize):
    w1, w2 = bigram
    bigramCount = bigramFreq[bigram] + 1
    unigramCount = unigramFreq[w1] + vocabSize
    return bigramCount / unigramCount

def addKSmoothingUnigram(word, unigramFreq, totalWords, vocabSize, k=0.5):
    return (unigramFreq.get(word, 0) + k) / (totalWords + k * vocabSize)

def perplexity_calculation(tokens,unigramFreq, bigramFreq, vocabSize,n = 2 ):
    log_odds = 0
    N = len(tokens)

    for i in range(1,N):
        if n == 2:
            bigram = (tokens[i-1], tokens[i])
            bigram_prob = laplaceSmoothingBigram(bigram, bigramFreq, unigramFreq, vocabSize)
        elif n == 1:
            unigram = tokens[i]
            unigram_prob = laplaceSmoothingUnigram(unigram, unigramFreq, totalWords, vocabSize)
        else: 
            raise ValueError("Must be a bigram or unigram")
        
        log_odds += math.log(bigram_prob)
    
    perplxity = math.exp(-log_odds/N)
    return perplxity


# Uncomment the lines below to read from train.txt
# preprocessed_lines = []
# with open('train.txt', 'r') as file:
#     for line in file:
#         cleaned_line = preprocess_text(line)
#         preprocessed_lines.append(cleaned_line)
# tokens = [token for line in preprocessed_lines for token in line.split()]

# Test sentence
tokens = ['the', 'dog', 'barked', 'at', 'the', 'cat', 'and', 'the', 'dog', 'ran']

# Replace low-frequency words with <UNK>
processedTokens = replaceWithUnk(tokens, minThreshold=1)
print("Tokens after replacing low-frequency words with <UNK>:", processedTokens)

# Frequency distributions for unigrams and bigrams
unigramFreq = FreqDist(processedTokens)
bigramFreq = FreqDist(bigrams(processedTokens))

# Calculate total words and vocabulary size
totalWords = len(processedTokens)
vocabSize = len(set(unigramFreq))

print("\nTotal Words:", totalWords)
print("Vocabulary Size:", vocabSize)

# Laplace smoothed unigram probabilities
print("\nLaplace Smoothed Unigram Probabilities:")
for word in unigramFreq.keys():
    probLaplaceUnigram = laplaceSmoothingUnigram(word, unigramFreq, totalWords, vocabSize)
    print(f"P({word}) = {probLaplaceUnigram:.4f}")

# Laplace smoothed bigram probabilities
print("\nLaplace Smoothed Bigram Probabilities:")
for bigram in bigramFreq.keys():
    probLaplaceBigram = laplaceSmoothingBigram(bigram, bigramFreq, unigramFreq, vocabSize)
    print(f"P({bigram[1]} | {bigram[0]}) = {probLaplaceBigram:.4f}")


# Perplexity Calculation
perplexity_bigram = perplexity_calculation(processedTokens, unigramFreq, bigramFreq, vocabSize, n=2)
print("\nPerplexity for Bigram Model: ", perplexity_bigram)

perplexity_unigram = perplexity_calculation(processedTokens, unigramFreq, bigramFreq, vocabSize, n=1)
print("\nPerplexity for Unigram Model: ", perplexity_unigram)


# Add-k smoothed unigram probabilities (k=0.5 example)
print("\nAdd-0.5 Smoothed Unigram Probabilities:")
for word in unigramFreq.keys():
    probAddKUnigram = addKSmoothingUnigram(word, unigramFreq, totalWords, vocabSize, k=0.5)
    print(f"P({word}) = {probAddKUnigram:.4f}")


