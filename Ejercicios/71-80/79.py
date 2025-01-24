'''
Representa una cuenta bancaria con deposito y retiro 
debe haber un titular y un saldo
Utiliza POO
'''

class Cuenta:
    def __init__(self, nombre, saldo):
        self.nombre = nombre
        self.saldo = saldo
    
    def depositar(self, cantidad):
        self.saldo += cantidad
        print(f'Se depositaron {cantidad}, el saldo total es de: ${round(self.saldo, 2)}')
    
    def retirar(self, cantidad):
        if cantidad <= self.saldo:
            self.saldo -= cantidad
            print(f'Se retiraron {cantidad}, el saldo total es de: ${round(self.saldo, 2)}')
        else:
            print(f'Imposible hacer el retiro, no cuenta con tanto dinero')
    
    def mostrar(self):
        print(self.__dict__)

cliente = Cuenta('Kikin', 1000)
cliente.depositar(100)
cliente.mostrar()
cliente.retirar(1200)
cliente.mostrar()

