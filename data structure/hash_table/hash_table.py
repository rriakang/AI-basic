
hash_table = list([0 for i in range(8)])


print(hash_table)

def get_key(data):
    return hash(data)

get_key('rira')
# 해쉬 함수 생성
def hash_func(key) :
    return key % 8

def save_data(data,value):
    hash_address = hash_func(get_key(data))
    hash_table[hash_address] = value

def read_data(data):
    hash_address = hash_func(get_key(data))
    print(hash_table[hash_address])
    
    
    

save_data('rr','01012345678')
save_data('aa','01056781234')

read_data('rr')
read_data('aa')

print(hash_table)
