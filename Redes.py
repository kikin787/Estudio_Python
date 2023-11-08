import socket
import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Configuración del servidor
server_ip = '127.0.0.1'  # Dirección IP del servidor
server_port = 12345     # Puerto en el que escuchará el servidor

# Generar una clave aleatoria para cifrado AES
encryption_key = get_random_bytes(16)

# Crear un diccionario para mantener las sesiones de los clientes
client_sessions = {}

# Inicializar el servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)

print(f"Servidor en {server_ip}:{server_port}")

# Función para otorgar una nueva sesión a un cliente
def grant_new_session(client_socket):
    # Generar un ID de cliente único
    client_id = random.randint(1, 1000)

    # Enviar el ID del cliente al cliente
    client_socket.send(str(client_id).encode())

    # Enviar la clave de cifrado al cliente
    client_socket.send(encryption_key)

    # Enviar la lista de clientes conectados al cliente
    connected_clients = list(client_sessions.keys())
    client_socket.send(str(connected_clients).encode())

    # Agregar la sesión del cliente al diccionario
    client_sessions[client_id] = client_socket

# Esperar conexiones entrantes
while True:
    client_socket, client_address = server_socket.accept()
    print(f"Conexión entrante desde {client_address}")
    grant_new_session(client_socket)
