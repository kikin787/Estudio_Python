'''
verificar si una palabra es palindromo usando lambda
'''

palindromo = lambda x: x == x[::-1]
print(palindromo('gato'))