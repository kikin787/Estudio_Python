import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Descargar recursos necesarios de NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configurar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar dataset
df = pd.read_csv('movie_data.csv')

# Preprocesamiento de texto
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Eliminar HTML
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Solo letras y espacios
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Aplicar preprocesamiento y guardar tokens
df['tokens'] = df['review'].apply(preprocess_text)
df['processed_review'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

# Crear vocabulario
def create_vocabulary(token_lists):
    word_counts = Counter()
    for tokens in token_lists:
        word_counts.update(tokens)
    return {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}

vocabulary = create_vocabulary(df['tokens'])
vocab_size = len(vocabulary)
embedding_dim = 100  # Dimensión de los embeddings

# ===============================================================
# IMPLEMENTACIÓN DEL MODELO SKIP-GRAM CON NÚMERO FIJO DE PARES POR CRÍTICA
# ===============================================================
class SkipGramSampledDataset(Dataset):
    def __init__(self, token_lists, vocabulary, window_size=2, negative_samples=3, pairs_per_review=10, max_tokens=300):
        """
        token_lists: Lista de listas de tokens.
        vocabulary: Diccionario palabra -> id.
        window_size: Tamaño de la ventana.
        negative_samples: Número de muestras negativas por par positivo.
        pairs_per_review: Número de pares positivos a generar por crítica.
        max_tokens: Máximo de tokens a considerar por crítica.
        """
        self.token_lists = token_lists
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.pairs_per_review = pairs_per_review
        self.vocab_size = len(vocabulary)
        self.max_tokens = max_tokens
        # Número fijo de pares por crítica:
        self.fixed_pairs = self.pairs_per_review * (1 + self.negative_samples)

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        if self.max_tokens is not None:
            tokens = tokens[:self.max_tokens]
        token_ids = [self.vocabulary[token] for token in tokens if token in self.vocabulary]
        n = len(token_ids)
        centers = []
        contexts = []
        labels = []
        # Generar EXACTAMENTE pairs_per_review iteraciones
        for _ in range(self.pairs_per_review):
            if n == 0:
                # Si no hay tokens, rellenar con ceros
                centers.extend([0] * (1 + self.negative_samples))
                contexts.extend([0] * (1 + self.negative_samples))
                labels.extend([0] * (1 + self.negative_samples))
            else:
                i = np.random.randint(0, n)
                center = token_ids[i]
                # Calcular el rango de contexto
                start = max(0, i - self.window_size)
                end = min(n, i + self.window_size + 1)
                context_indices = [j for j in range(start, end) if j != i]
                if context_indices:
                    j = np.random.choice(context_indices)
                    context = token_ids[j]
                    centers.append(center)
                    contexts.append(context)
                    labels.append(1)
                else:
                    centers.append(center)
                    contexts.append(center)
                    labels.append(0)
                # Muestras negativas
                for _ in range(self.negative_samples):
                    negative = np.random.randint(0, self.vocab_size)
                    centers.append(center)
                    contexts.append(negative)
                    labels.append(0)
        # Asegurarse de que siempre se retorne el mismo número de pares
        assert len(centers) == self.fixed_pairs, f"{len(centers)} != {self.fixed_pairs}"
        return (torch.tensor(centers, dtype=torch.long),
                torch.tensor(contexts, dtype=torch.long),
                torch.tensor(labels, dtype=torch.float))

# Crear dataset y DataLoader
sampled_dataset = SkipGramSampledDataset(df['tokens'], vocabulary, window_size=2, negative_samples=3, pairs_per_review=10, max_tokens=300)
batch_size_sg = 256
skipgram_loader = DataLoader(sampled_dataset, batch_size=batch_size_sg, shuffle=True, num_workers=4)

# Modelo SKIP-GRAM modificado para trabajar con tensores 2D
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words):
        # center_words y context_words tienen forma (batch, num_pairs)
        center_embeds = self.in_embeddings(center_words)   # (batch, num_pairs, embedding_dim)
        context_embeds = self.out_embeddings(context_words)  # (batch, num_pairs, embedding_dim)
        # Producto punto a lo largo de la dimensión de embedding
        scores = torch.sum(center_embeds * context_embeds, dim=2)  # (batch, num_pairs)
        return scores

skipgram_model = SkipGramModel(vocab_size, embedding_dim).to(device)
skipgram_criterion = nn.BCEWithLogitsLoss()
skipgram_optimizer = optim.Adam(skipgram_model.parameters(), lr=0.001)

# Entrenamiento del modelo SKIP-GRAM
skipgram_epochs = 5
for epoch in range(skipgram_epochs):
    total_loss = 0
    for center, context, label in skipgram_loader:
        # Saltar lotes vacíos
        if center.nelement() == 0:
            continue
        center = center.to(device)      # Forma: (batch, fixed_pairs)
        context = context.to(device)
        label = label.to(device)        # Forma: (batch, fixed_pairs)
        skipgram_optimizer.zero_grad()
        outputs = skipgram_model(center, context)  # (batch, fixed_pairs)
        loss = skipgram_criterion(outputs, label)
        loss.backward()
        skipgram_optimizer.step()
        total_loss += loss.item()
    print(f"SkipGram Epoch {epoch+1}/{skipgram_epochs}, Loss: {total_loss/len(skipgram_loader):.4f}")

# Extraer la matriz de embeddings entrenada
embeddings = skipgram_model.in_embeddings.weight.data.cpu().numpy()

# Función para representar cada crítica como el promedio de los embeddings de sus palabras
def get_review_embedding(tokens, vocabulary, embeddings):
    token_ids = [vocabulary[token] for token in tokens if token in vocabulary]
    if len(token_ids) == 0:
        return np.zeros(embedding_dim)
    else:
        return np.mean(embeddings[token_ids], axis=0)

df['review_embedding'] = df['tokens'].apply(lambda tokens: get_review_embedding(tokens, vocabulary, embeddings))

# ===============================================================
# MODELO DE RED NEURONAL (NNBP) PARA CLASIFICACIÓN DE SENTIMIENTOS
# ===============================================================
X = np.vstack(df['review_embedding'].values)
y = df['sentiment'].values  # 0 o 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size_cls = 32
train_dataset = ReviewDataset(X_train_tensor, y_train_tensor)
test_dataset = ReviewDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size_cls, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_cls, shuffle=False)

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

hidden_dim = 64
classifier = NNBPClassifier(embedding_dim, hidden_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

classifier_epochs = 10
for epoch in range(classifier_epochs):
    classifier.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = classifier(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Classifier Epoch {epoch+1}/{classifier_epochs}, Loss: {total_loss/len(train_loader):.4f}')

classifier.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = classifier(batch_X)
        preds = (outputs >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_true.append(batch_y.cpu().numpy())
y_pred_nn = np.vstack(all_preds)
y_true = np.vstack(all_true)
accuracy_nn = accuracy_score(y_true, y_pred_nn)
print(f'Neural Network Accuracy: {accuracy_nn:.4f}')
print(classification_report(y_true, y_pred_nn))
cm_nn = confusion_matrix(y_true, y_pred_nn)
tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()
metrics_nn = {
    'Accuracy': accuracy_nn,
    'Specificity': tn_nn / (tn_nn + fp_nn) if (tn_nn+fp_nn)>0 else 0,
    'Sensitivity': tp_nn / (tp_nn + fn_nn) if (tp_nn+fn_nn)>0 else 0,
    'Precision': precision_score(y_true, y_pred_nn),
    'False Positive Rate': fp_nn / (fp_nn + tn_nn) if (fp_nn+tn_nn)>0 else 0,
    'False Negative Rate': fn_nn / (fn_nn + tp_nn) if (fn_nn+tp_nn)>0 else 0,
    'Positive Predictive Value': tp_nn / (tp_nn + fp_nn) if (tp_nn+fp_nn)>0 else 0,
    'Negative Predictive Value': tn_nn / (tn_nn + fn_nn) if (tn_nn+fn_nn)>0 else 0
}

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.4f}')
print(classification_report(y_test, y_pred_lr))
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
metrics_lr = {
    'Accuracy': accuracy_lr,
    'Specificity': tn_lr / (tn_lr + fp_lr) if (tn_lr+fp_lr)>0 else 0,
    'Sensitivity': tp_lr / (tp_lr + fn_lr) if (tp_lr+fn_lr)>0 else 0,
    'Precision': precision_score(y_test, y_pred_lr),
    'False Positive Rate': fp_lr / (fp_lr + tn_lr) if (fp_lr+tn_lr)>0 else 0,
    'False Negative Rate': fn_lr / (fn_lr + tp_lr) if (fn_lr+tp_lr)>0 else 0,
    'Positive Predictive Value': tp_lr / (tp_lr + fp_lr) if (tp_lr+fp_lr)>0 else 0,
    'Negative Predictive Value': tn_lr / (tn_lr + fn_lr) if (tn_lr+fn_lr)>0 else 0
}

metrics_names = list(metrics_nn.keys())
nn_values = list(metrics_nn.values())
lr_values = [metrics_lr[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, nn_values, width, label='Neural Network', color='skyblue')
plt.bar(x + width/2, lr_values, width, label='Logistic Regression', color='salmon')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparative Performance Metrics')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()
for i, v in enumerate(nn_values):
    plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
for i, v in enumerate(lr_values):
    plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\nDetailed Metrics for Neural Network Model:")
for metric, value in metrics_nn.items():
    print(f"{metric}: {value:.3f}")

print("\nDetailed Metrics for Logistic Regression Model:")
for metric, value in metrics_lr.items():
    print(f"{metric}: {value:.3f}")
