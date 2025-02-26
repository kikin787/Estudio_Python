import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from datetime import datetime

class EmotionNNBP:
    def __init__(self):
        self.emotions = ['feliz', 'enojado', 'triste', 'sorprendido', 'neutral']
        self.data_path = 'dataset'
        self.model_path = 'emotion_model_nnbp.npy'
        self.input_size = 48 * 48
        self.hidden_size = 256  # Aumentado para mejor capacidad de aprendizaje
        self.output_size = len(self.emotions)
        self.learning_rate = 0.001  # Reducido para mayor estabilidad
        self.epochs = 100
        self.batch_size = 32

        # Inicialización de pesos con Xavier/Glorot
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def sigmoid(self, x):
        """Función de activación sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def create_dataset_structure(self):
        os.makedirs(self.data_path, exist_ok=True)
        for emotion in self.emotions:
            os.makedirs(os.path.join(self.data_path, emotion), exist_ok=True)

    def capture_dataset(self):
        self.create_dataset_structure()
        cap = cv2.VideoCapture(0)

        for emotion in self.emotions:
            count = 0
            print(f"\nPreparado para capturar: {emotion}")
            input(f"Presiona Enter para comenzar a capturar {emotion}...")
            print(f"Capturando {emotion}... Presiona 'q' cuando termines.")

            while count < 140:  # Límite de 140 imágenes por emoción
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    file_path = os.path.join(self.data_path, emotion, f'{emotion}_{timestamp}.jpg')
                    cv2.imwrite(file_path, face_roi)
                    count += 1
                    print(f"Capturada imagen {count}/140 para {emotion}")

                cv2.putText(frame, f"{emotion}: {count}/140", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Capturando Dataset', frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or count >= 140:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def load_data(self):
        """Carga y preprocesa las imágenes del dataset"""
        X = []
        y = []

        # Verificar si existe el directorio del dataset
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró el directorio del dataset: {self.data_path}")

        print("Cargando dataset...")
        for idx, emotion in enumerate(self.emotions):
            emotion_path = os.path.join(self.data_path, emotion)

            # Verificar si existe el directorio de la emoción
            if not os.path.exists(emotion_path):
                print(f"Advertencia: No se encontró el directorio para {emotion}")
                continue

            # Obtener lista de imágenes
            image_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"Advertencia: No se encontraron imágenes para {emotion}")
                continue

            print(f"Cargando imágenes de {emotion}: {len(image_files)} encontradas")

            for img_file in image_files:
                img_path = os.path.join(emotion_path, img_file)
                try:
                    # Leer y preprocesar imagen
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Error al cargar imagen: {img_path}")
                        continue

                    # Asegurar que la imagen sea 48x48
                    img = cv2.resize(img, (48, 48))

                    # Aplanar y normalizar la imagen
                    img_flat = img.flatten() / 255.0

                    X.append(img_flat)
                    y.append(idx)

                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
                    continue

        if not X or not y:
            raise ValueError("No se pudieron cargar imágenes. Asegúrate de capturar el dataset primero.")

        X = np.array(X)
        y = np.array(y)

        # Convertir etiquetas a one-hot encoding
        y = to_categorical(y, num_classes=len(self.emotions))

        print(f"\nDataset cargado exitosamente:")
        print(f"Tamaño del dataset: {len(X)} imágenes")
        print(f"Dimensiones de entrada: {X.shape}")
        print(f"Dimensiones de salida: {y.shape}")

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nConjunto de entrenamiento: {X_train.shape[0]} imágenes")
        print(f"Conjunto de prueba: {X_test.shape[0]} imágenes")

        return X_train, X_test, y_train, y_test

    

    def save_model(self):
        """Guarda el modelo en formato .npz"""
        np.savez(self.model_path.replace('.npy', '.npz'),
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_model(self):
        """Carga el modelo desde formato .npz"""
        try:
            model_data = np.load(self.model_path.replace('.npy', '.npz'))
            self.W1 = model_data['W1']
            self.b1 = model_data['b1']
            self.W2 = model_data['W2']
            self.b2 = model_data['b2']
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def train(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return

        print("Iniciando entrenamiento...")
        losses = []
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            total_loss = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Forward pass
                Z1 = np.dot(X_batch, self.W1) + self.b1
                A1 = self.sigmoid(Z1)
                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self.sigmoid(Z2)

                # Backward pass
                loss = np.mean((y_batch - A2) ** 2)
                total_loss += loss

                error_output = (y_batch - A2) * self.sigmoid_derivative(A2)
                error_hidden = np.dot(error_output, self.W2.T) * self.sigmoid_derivative(A1)

                # Update weights and biases
                self.W2 += self.learning_rate * np.dot(A1.T, error_output)
                self.b2 += self.learning_rate * np.sum(error_output, axis=0, keepdims=True)
                self.W1 += self.learning_rate * np.dot(X_batch.T, error_hidden)
                self.b1 += self.learning_rate * np.sum(error_hidden, axis=0, keepdims=True)

            avg_loss = total_loss / (len(X_train) / self.batch_size)
            losses.append(avg_loss)

            if epoch % 5 == 0:
                print(f'Época {epoch}/{self.epochs} - Pérdida: {avg_loss:.4f}')

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Guardar mejor modelo
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping activado")
                    break

        self.plot_training_loss(losses)
        print("Entrenamiento completado")

    def plot_training_loss(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Evolución de la pérdida en el entrenamiento')
        plt.grid(True)
        plt.savefig('training_loss_nnbp.png')
        plt.close()

    def predict_emotion(self, face_roi):
        face_roi = face_roi.flatten() / 255.0
        Z1 = np.dot(face_roi, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return np.argmax(A2), np.max(A2)

    def detect_emotion(self):
        if not self.load_model():
            print("Por favor, entrena el modelo primero.")
            return

        cap = cv2.VideoCapture(0)
        print("Presiona 'q' para salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                emotion_idx, confidence = self.predict_emotion(face_roi)
                emotion = self.emotions[emotion_idx]

                # Dibujar rectángulo y etiqueta
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion} ({confidence:.1%})"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 255, 0), 2)

            cv2.imshow('Detector de Emociones', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    modelo = EmotionNNBP()
    while True:
        print("\nSistema de Reconocimiento de Emociones NNBP")
        print("============================================")
        print("0. Salir")
        print("1. Capturar dataset")
        print("2. Entrenar modelo")
        print("3. Detectar emociones en tiempo real")

        try:
            opcion = int(input("\nElige una opción (0-3): "))

            if opcion == 0:
                print("¡Hasta luego!")
                break
            elif opcion == 1:
                modelo.capture_dataset()
            elif opcion == 2:
                modelo.train()
            elif opcion == 3:
                if not os.path.exists(modelo.model_path.replace('.npy', '.npz')):
                    print("Error: No se encontró el modelo entrenado.")
                    print("Por favor, entrena el modelo primero (opción 2).")
                else:
                    modelo.detect_emotion()
            else:
                print("Opción no válida. Por favor, elige una opción entre 0 y 3.")
        except ValueError:
            print("Por favor, ingresa un número válido.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()