import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    print(text)
    

    words = text.split(' ')
    
    words_to_id = {}
    id_to_words = {}
    
    for word in words :
        if word not in words_to_id :
            new_id = len(words)
            words_to_id[words] = new_id
            id_to_words[new_id] = words
            
            
    corpus = np.array([words_to_id[w] for w in words])
    
    return(corpus,words_to_id,id_to_words)


# co-occurence_matrix
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size), dtype=np.int32) # 0으로 채워진 2차원 배열 초기화
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1): 
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0 :
                left_word_id = corpus[left_idx]
                co_matrix[word_id,left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1
    return co_matrix


# def cos_similartiy(x,y):
#     nx = x /  np.sqrt(np.sum(x**2)) # x의 정규화
#     ny = y / np.sqrt(np.sum(y**2)) # y의 정규화
    
    
#     return np.dot(nx,ny)


# 인수로 제로벡터(원소가 모두 0 인 벡터)가 들어오면 '0으로 나누기' 오류가 발생해버림


def cos_similartiy(x,y, eps=1e-8):
    nx = x /  (np.sqrt(np.sum(x**2))+ eps) # x의 정규화
    ny = y / (np.sqrt(np.sum(y**2)) + eps) # y의 정규화
    
    
    return np.dot(nx,ny)


# 유사 단어의 랭킹 표시


# query : 검색어
# word_to_id : 단어에서 단어 ID로의 딕셔너ㅣㄹ
# id_to_word : 단어 ID에서 단어로의 딕셔너리
# word_matrix : 단어 벡터들을 한데 모은 행렬. 각 행에는 대응하는 단어의 벡터가 저장되어 있다고 가정함
# top : 상위 몇개까지 출력할지 설정

def most_similar(query,word_to_id,id_to_word,word_matrix, top=5) :
    # 검색어를 꺼낸다
    if query not in word_to_id :
        print("%s(을)를 찾을 수 없습니다." %query)
        return
    
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    
    # 코사인 유사도 계산
    
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    
    for i in range(vocab_size):
        similarity[i] = cos_similartiy(word_matrix[i], query_vec)
        
        
        
    # 코사인 유사도를 기준으로 내림차순으로 출력
    
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        
        print(' %s: %s' % (id_to_word[i].similarity[i]))
        
        count += 1
        if count >= top:
            return
        
        

# C  : 동시발생 행렬
# verbose : 진행상황 출력 여버룰 결정하는 플래그    
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / S[j] * S[i] + eps)
            M[i,j] = max(0,pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
                    
                    
    return M