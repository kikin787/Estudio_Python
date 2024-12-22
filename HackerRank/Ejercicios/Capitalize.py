def solve(s):
    palabras = s.split(' ')
    nsM = [palabra.capitalize() if palabra else '' for palabra in palabras]
    resultado = ' '.join(nsM)
    return resultado       

if __name__ == '__main__':

    s = input()
    print(solve(s))

