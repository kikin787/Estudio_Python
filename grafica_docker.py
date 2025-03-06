import networkx as nx
import matplotlib.pyplot as plt

# Crear el grafo
G = nx.DiGraph()

# Agregar nodos (contenedores y host)
G.add_node("Host (localhost)", color="lightgreen", size=3000, description="Punto de acceso desde el navegador")
G.add_node("iTop Container\n(172.18.0.4)", color="lightblue", size=3000, description="Contenedor principal de iTop")
G.add_node("MariaDB Container\n(172.18.0.2)", color="lightcoral", size=3000, description="Base de datos MariaDB")
G.add_node("phpMyAdmin Container\n(172.18.0.3)", color="lightyellow", size=3000, description="Interfaz de administración de la base de datos")

# Agregar conexiones (aristas) con etiquetas
G.add_edge("Host (localhost)", "iTop Container\n(172.18.0.4)", label="HTTPS (443)")
G.add_edge("iTop Container\n(172.18.0.4)", "MariaDB Container\n(172.18.0.2)", label="MySQL (3306)")
G.add_edge("Host (localhost)", "phpMyAdmin Container\n(172.18.0.3)", label="HTTP (8080)")
G.add_edge("phpMyAdmin Container\n(172.18.0.3)", "MariaDB Container\n(172.18.0.2)", label="MySQL (3306)")

# Obtener atributos de nodos para personalización
node_colors = [G.nodes[node].get("color", "lightblue") for node in G.nodes]
node_sizes = [G.nodes[node].get("size", 1000) for node in G.nodes]

# Dibujar el grafo
pos = nx.spring_layout(G, seed=42)  # Layout para posicionar los nodos
nx.draw(
    G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, font_weight="bold"
)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Configurar título y mostrar la gráfica
plt.title("Docker Network Diagram: iTop System")
plt.show()