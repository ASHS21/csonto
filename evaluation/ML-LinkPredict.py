import pandas as pd
import networkx as nx
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the graph data
df = pd.read_csv('/Users/alialmoharif/Desktop/CSKG-FYP.csv')

# Data preprocessing
df['source_name'] = df['Source'].apply(lambda x: re.search(r'name: "([^"]+)"', x).group(1) if re.search(r'name: "([^"]+)"', x) else 'DefaultName')
df['target_name'] = df['Target'].apply(lambda x: re.search(r'name: "([^"]+)"', x).group(1) if re.search(r'name: "([^"]+)"', x) else 'DefaultName')
df['source_score'] = df['Source'].apply(lambda x: int(re.search(r'score: (\d+)', x).group(1)) if re.search(r'score: (\d+)', x) else 0)
df['relation_type'] = df['Relations'].apply(lambda x: re.search(r'\[:(\w+)\]', x).group(1) if re.search(r'\[:(\w+)\]', x) else 'Unknown')
df['relation_type'] = df['relation_type'].astype('category').cat.codes
df.drop(columns=['Source', 'Target', 'Relations'], inplace=True)

# Create a graph from the DataFrame
G = nx.from_pandas_edgelist(df, source='source_name', target='target_name', edge_attr=True, create_using=nx.DiGraph())
G_undirected = G.to_undirected()

# Generate positive and negative examples
positive_examples = pd.DataFrame({
    'source': [u for u, v in G.edges()],
    'target': [v for u, v in G.edges()],
    'label': 1
})

def generate_negative_examples(graph, num_examples):
    negative_edges = []
    nodes = list(graph.nodes())
    attempted_pairs = set()

    while len(negative_edges) < num_examples:
        u, v = random.sample(nodes, 2)
        if (u, v) not in attempted_pairs and (v, u) not in attempted_pairs:
            if not graph.has_edge(u, v) and not graph.has_edge(v, u):
                negative_edges.append((u, v, 0))  # Append as a tuple with the label 0
            attempted_pairs.add((u, v))
            attempted_pairs.add((v, u))

    return negative_edges

negative_examples = pd.DataFrame(generate_negative_examples(G_undirected, len(G.edges())), columns=['source', 'target', 'label'])

# Combine positive and negative examples, shuffle them
df = pd.concat([positive_examples, negative_examples]).sample(frac=1).reset_index(drop=True)

# Feature engineering function
def compute_graph_features(row):
    u, v = row['source'], row['target']
    features = {
        'common_neighbors': len(list(nx.common_neighbors(G_undirected, u, v))),
        'jaccard_coefficient': next(nx.jaccard_coefficient(G_undirected, [(u, v)]), (None, None, 0))[2],
        'adamic_adar_index': next(nx.adamic_adar_index(G_undirected, [(u, v)]), (None, None, 0))[2],
        'preferential_attachment': next(nx.preferential_attachment(G_undirected, [(u, v)]), (None, None, 0))[2],
        'degree_u': G.degree(u),
        'degree_v': G.degree(v)
    }
    return features

# Apply feature engineering
features_list = [compute_graph_features(row) for index, row in df.iterrows()]
feature_df = pd.DataFrame(features_list)
feature_df['label'] = df['label']

# Now, move to the machine learning part
X_train, X_test, y_train, y_test = train_test_split(feature_df.drop('label', axis=1), feature_df['label'], test_size=0.2, random_state=42)

# Model training with hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2']}
model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
model.fit(X_train, y_train)


# Model evaluation
best_model = model.best_estimator_
predictions = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, predictions))
print("Best Model Precision:", precision_score(y_test, predictions))
print("Best Model Recall:", recall_score(y_test, predictions))
print("Best Model F1 Score:", f1_score(y_test, predictions))

# ROC Curve computation
probabilities = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
print("Best Model ROC AUC:", roc_auc)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probabilities)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision_points, recall_points, _ = precision_recall_curve(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

fig, ax = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns to fit all metrics and plots in an appropriate layout

# Plotting each metric in its respective subplot
metrics = [accuracy, precision, recall, f1]
names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['skyblue', 'salmon', 'lightgreen', 'gold']

for i, metric in enumerate(metrics):
    ax[0, i].bar(names[i], metric, color=colors[i])
    ax[0, i].set_title(names[i])
    ax[0, i].set_ylim(0, 1)  # Set y-axis limit to scale between 0 and 1

# ROC Curve
ax[1, 0].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='darkorange')
ax[1, 0].plot([0, 1], [0, 1], linestyle='--', color='navy')
ax[1, 0].set_xlim([0.0, 1.0])
ax[1, 0].set_ylim([0.0, 1.05])
ax[1, 0].set_xlabel('False Positive Rate')
ax[1, 0].set_ylabel('True Positive Rate')
ax[1, 0].set_title('Receiver Operating Characteristic')
ax[1, 0].legend(loc="lower right")

# Precision-Recall Curve
ax[1, 1].plot(recall_points, precision_points, color='blue', label=f'PR curve (area = {pr_auc:.2f})')
ax[1, 1].set_xlabel('Recall')
ax[1, 1].set_ylabel('Precision')
ax[1, 1].set_title('Precision-Recall curve')
ax[1, 1].legend(loc="lower left")

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax[1, 2], cmap=plt.cm.Blues)
ax[1, 2].set_title('Confusion Matrix')

# Adjust the layout for visibility
ax[1, 3].axis('off')  # Turn off the 4th subplot in the second row as it's unused

plt.tight_layout()
plt.show()
