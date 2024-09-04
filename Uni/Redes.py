import socket
import threading

# Configuración del servidor
server_ip = '10.1.1.1'
server_port = 1212

# Inicializar el servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)

print(f"Servidor en {server_ip}:{server_port}")

# Diccionario para mantener las sesiones de los clientes
client_sessions = {}

# Función para manejar las conexiones de los clientes
def handle_client(client_socket, client_address):
    # Otorgar nueva sesión al cliente
    client_id = len(client_sessions) + 1

    # Agregar la sesión del cliente al diccionario
    client_sessions[client_id] = client_socket

    print(f"Cliente {client_id} conectado desde {client_address}")

    # Manejar mensajes entrantes de este cliente
    while True:
        try:
            message = client_socket.recv(1024)
            if not message:
                break

            print(f"Mensaje del Cliente {client_id}: {message.decode()}")

            # Reenviar mensaje a otros clientes
            for other_client_id, other_client_socket in client_sessions.items():
                if other_client_id != client_id:
                    try:
                        other_client_socket.send(message)
                    except Exception as e:
                        print(f"Error al reenviar mensaje al Cliente {other_client_id}: {e}")

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
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_thread.start()

# Iniciar hilo para aceptar conexiones entrantes
accept_thread = threading.Thread(target=accept_connections)
accept_thread.start()