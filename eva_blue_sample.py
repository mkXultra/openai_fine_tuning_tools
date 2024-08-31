import nltk
from nltk.translate.bleu_score import sentence_bleu

# NLTKのダウンロード（初回のみ必要）
nltk.download('punkt')

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

# 英語の例
en_reference = ['this', 'is', 'a', 'test']
en_candidate = ['this', 'is', 'a', 'test']
en_score = calculate_bleu(en_reference, en_candidate)
print(f'English BLEU score: {en_score}')

# 日本語の例（文字単位で分割）
ja_reference = list('これはテストです')
ja_candidate = list('これはてすとです')
ja_score = calculate_bleu(ja_reference, ja_candidate)
print(f'Japanese BLEU score: {ja_score}')

# 日本語の別の例
ja_reference2 = list('吾輩は猫である')
ja_candidate2 = list('我輩は犬である')
ja_score2 = calculate_bleu(ja_reference2, ja_candidate2)
print(f'Japanese BLEU score 2: {ja_score2}')