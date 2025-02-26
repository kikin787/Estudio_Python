import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar dataset
df = pd.read_csv('movie_data.csv')

def preprocess_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()  # Eliminar HTML
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres especiales
    tokens = word_tokenize(text)  # Tokenización
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Eliminar stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lematización
    return ' '.join(tokens)

# Aplicar preprocesamiento
df['processed_review'] = df['review'].apply(preprocess_text)

# Crear vocabulario
def create_vocabulary(processed_texts):
    word_counts = Counter()
    for text in processed_texts:
        word_counts.update(text.split())
    return {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}

vocabulary = create_vocabulary(df['processed_review'])
vocab_size = len(vocabulary)
embedding_dim = 100

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['processed_review']).toarray()
y = df['sentiment'].values

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Convertir a tensores y mover a GPU si está disponible
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Modelo NNBP
class NNBPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NNBPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Inicializar modelo y mover a GPU
hidden_dim = 64
model = NNBPClassifier(X_train.shape[1], hidden_dim).to(device)

# Definir función de pérdida y optimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluación
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred >= 0.5).float()
    accuracy = accuracy_score(y_test_tensor.cpu(), y_pred_labels.cpu())
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test_tensor.cpu(), y_pred_labels.cpu()))

# Calculate metrics for Neural Network model
y_pred_nn = (y_pred >= 0.5).float().cpu().numpy()
y_true = y_test_tensor.cpu().numpy()

cm_nn = confusion_matrix(y_true, y_pred_nn)
tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()

metrics_nn = {
    'Accuracy': accuracy_score(y_true, y_pred_nn),
    'Specificity': tn_nn / (tn_nn + fp_nn),
    'Sensitivity': tp_nn / (tp_nn + fn_nn),
    'Precision': precision_score(y_true, y_pred_nn),
    'False Positive Rate': fp_nn / (fp_nn + tn_nn),
    'False Negative Rate': fn_nn / (fn_nn + tp_nn),
    'Positive Predictive Value': tp_nn / (tp_nn + fp_nn),
    'Negative Predictive Value': tn_nn / (tn_nn + fn_nn)
}

# Create comparative visualization
metrics_names = list(metrics_nn.keys())
nn_values = list(metrics_nn.values())

# Create the plot
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics_names))
width = 0.35

plt.bar(x, nn_values, width, label='Neural Network', color='skyblue')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Performance Metrics for Neural Network Classifier')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()

# Add value labels on top of each bar
for i, v in enumerate(nn_values):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print metrics in a formatted way
print("\nDetailed Metrics for Neural Network Model:")
for metric, value in metrics_nn.items():
    print(f"{metric}: {value:.3f}")

# Create confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()