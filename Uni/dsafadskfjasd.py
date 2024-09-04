from pydoc import cli
import socket
import os
import threading
import json
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


# Configuración del servidor
server_ip = "10.1.1.1"
server_port = 1212

# Inicializar el servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)

print(f"Servidor en {server_ip}:{server_port}")

# Diccionario para mantener las sesiones de los clientes y sus claves
client_sessions = {}


# Función para enviar información de clientes conectados a un nuevo cliente
def send_client_info(client_socket, client_id, client_key):
    client_info = {
        "client_id": client_id,
        "client_key": client_key,
        "client_list": [
            (client_id, client_key)
            for client_id, (client_socket, client_key) in client_sessions.items()
        ],
    }
    client_socket.sendall(json.dumps(client_info).encode())


def cifrar_mensaje(message, key):
    # Clave y vector de inicialización para el cifrado AES
    key = bytes.fromhex(key)
    iv = os.urandom(16)  # Vector de inicialización aleatorio

    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Cifra el mensaje y añade el IV al inicio del mensaje cifrado
    ciphertext = encryptor.update(message.encode()) + encryptor.finalize()

    return iv + ciphertext


def descifrar_mensaje(encrypted_message, key):
    key = bytes.fromhex(key)
    # Obtiene el IV del mensaje cifrado
    iv = encrypted_message[:16]
    ciphertext = encrypted_message[16:]

    # Crea un objeto Cipher y decryptor
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())

    decryptor = cipher.decryptor()

    # Descifra el mensaje
    decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()
    return decrypted_message.decode()


def handle_client(client_socket, client_address):
    # Otorgar nueva sesión y clave al cliente
    client_id = len(client_sessions) + 1
    client_key = secrets.token_hex(16)  # Generar una clave única

    # Almacena la sesión del cliente en el diccionario con su clave
    client_sessions[client_id] = (client_socket, client_key)

    print(f"Cliente {client_id} conectado desde {client_address}, clave: {client_key}")

    # Enviar información de clientes conectados al nuevo cliente
    send_client_info(client_socket, client_id, client_key)

    # Manejar mensajes entrantes de este cliente
    while True:
        try:
            encrypted_message = client_socket.recv(1024)
            if not encrypted_message:
                break

            # Descifra el mensaje

            decrypted_message = descifrar_mensaje(encrypted_message, client_key)

            # Añade la información del remitente al mensaje
            message_with_info = {
                "client_id": client_id,
                "client_key": client_key,
                "message": decrypted_message,
            }
            print(f"Mensaje del Cliente {client_id}: {decrypted_message}")
            # Reenviar mensaje a otros clientes
            for other_client_id, (other_client_socket, _) in client_sessions.items():
                if other_client_id != client_id:
                    try:
                        # Cifra el mensaje antes de enviarlo a otros clientes
                        other_client_socket.send(
                            cifrar_mensaje(
                                json.dumps(message_with_info),
                                client_sessions[other_client_id][1],
                            )
                        )
                    except Exception as e:
                        print(
                            f"Error al reenviar mensaje al Cliente {other_client_id}: {e}"
                        )

        except Exception as e:
            print(f"Error al recibir mensaje del Cliente {client_id}: {e}")
            break

    # Cerrar conexión con el cliente
    del client_sessions[client_id]
    client_socket.close()
    print(f"Cliente {client_id} desconectado")


# Función para manejar las conexiones entrantes
def accept_connections():
    while True:
        client_socket, client_address = server_socket.accept()

        # Iniciar un hilo para manejar la conexión del cliente
        client_thread = threading.Thread(
            target=handle_client, args=(client_socket, client_address)
        )
        client_thread.start()


# Iniciar hilo para aceptar conexiones entrantes
accept_thread = threading.Thread(target=accept_connections)
accept_thread.start()
