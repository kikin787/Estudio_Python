'''
Filtar numeros negativos de una lista utilizando filter
'''

na = [-11,1,2,3,-4,-1,-7,10]
ne = list(filter(lambda n: n < 0, na))
print(ne)