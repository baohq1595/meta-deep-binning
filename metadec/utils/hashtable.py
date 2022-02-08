import numpy as np
import hashlib

def simple_str2numeric(in_obj):
    if in_obj is None or in_obj == '':
        return None

    return int(hashlib.sha256(str(in_obj).encode('utf-8')).hexdigest(), 16)

def lmer2numeric(hash_str):
    '''
    A: 00
    T: 01
    G: 10
    C: 11
    '''
    val = 1
    for nucl in hash_str:
        val = val << 2
        if nucl == 'A':
            val |= 0
        elif nucl == 'T':
            val |= 1
        elif nucl == 'G':
            val |= 2
        elif nucl == 'C':
            val |= 3
        else:
            raise Exception('Bad nucliotide character!!!')

    return val

def numeric2lmer(numeric_val, expected_length=30):
    chars = []
    while(numeric_val > 1):
        mod = numeric_val % 4
        numeric_val = numeric_val // 4
        if mod == 3:
            char = 'C'
        elif mod == 2:
            char = 'G'
        elif mod == 1:
            char = 'T'
        elif mod == 0:
            char = 'A'
        else:
            raise Exception('Bad numeric value!!!')
        chars.append(char)

    if len(chars) != expected_length:
        raise Exception('Decoded length does not match with expected length!!!')

    return ''.join(list(reversed(chars)))

def tuple2numeric(tuple_val):
    

class Node:
    def __init__(self, key, value, encode_key):
        encoded_key = key
        self.key = encoded_key
        self.value = value
        self.next = None
        
#     def append(self, val):
#         if type(value) in (list, tuple):
#             self.values.extend(val)
#         else:
#             self.values.append(val)
            
    def __str__(self):
        return f'{self.key} -- {self.value}'


class LinkedList:
    def __init__(self, encode_key=True):
        self.head = None
        self.encode = encode_key
        
    def search(self, key):
        temp = self.head
        if temp is None:
            return None
        while temp:
            if temp.key == key:
                return temp
            temp = temp.next
        return None
        
    def insert(self, key, value):
        # Create new node, its next-pointer points to current head
        s_node = self.search(key)
        if s_node:
            s_node.value = value
        else:
            new_node = Node(key, value, self.encode)
            if self.head is not None:
                new_node.next = self.head
            # current head points to newly inserted note
            self.head = new_node
            
    def remove(self, key):
        temp = self.head
        prev = None
        if temp is None:
            return
        # Delete at the beginning
        if temp.key == key:
            self.head = temp.next
            val = temp.value
            return True
        
        # Other, traverse and delete
        while temp.next:
            # delete at the end
            if temp.next.next is None:
                if temp.next.key == key:
                    temp.next = None
                    return True
            else:
                if temp.key == key:
                    prev.next = temp.next
                    return True
            
            prev = temp
        
        return False
    
    def print_list(self):
        temp = self.head
        if not temp:
            print(None)
        while temp:
            if temp.next:
                print(temp.value,"--->",end="  ")
            else :
                print(temp.value)
            temp = temp.next

class Hashtable:
    def __init__(self, hash_size):
        self.size = hash_size
        self.hashmap = np.array([None]*self.size)
        
        # init each element to be a linkedlist
        for i in range(self.size):
            self.hashmap[i] = LinkedList()
            
    def hash_func(self, key):
        if type(key) is not int:
            # raise Exception('Key of hashmap should be in int')
            # convert to hash value
            key = simple_str2numeric(key)
            
        return key % self.size
            
    def insert(self, key, value):
        index = self.hash_func(key)
        self.hashmap[index].insert(key, value)
        
    def get(self, key, default_value=None):
        index = self.hash_func(key)
        node = self.hashmap[index].search(key)
        if node is None:
            if default_value is None:
                raise Exception(f'Key {key} not found.')
            else:
                return default_value
        else:
            return node.value
    
    def remove(self, key):
        index = self.hash_func(key)
        return self.hashmap[index].remove(key)
    
    def print_hash(self):
        print("Index \t\tValues\n")
        for x in range(self.size) :
            print(x,end="\t\t")
            self.hashmap[x].print_list()

if __name__ == "__main__":
    test_hashtab = Hashtable(10)
    test_hashtab.insert('aaa', 1)
    test_hashtab.insert('bbb', 4)
    test_hashtab.insert('bbb', 5)
    test_hashtab.insert('ccc', 7)
    test_hashtab.insert('ddd', 8)
    test_hashtab.insert('aaa', 11)

    test_hashtab.print_hash()
    test_hashtab.remove('aaa')
    test_hashtab.print_hash()
