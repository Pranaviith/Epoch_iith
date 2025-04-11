import numpy as np

data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

def preprocess_data(data):
    label_map = {label: i for i, label in enumerate(sorted(set(r[3] for r in data)))}
    X = np.array([[r[0], r[1], r[2]] for r in data], dtype=float)
    y = np.array([label_map[r[3]] for r in data], dtype=int)
    return X, y, list(label_map.keys())

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.class_names = None
    
    def fit(self, X, y, feature_names=None, class_names=None):
        self.feature_names = feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        self.class_names = class_names or [str(i) for i in range(len(np.unique(y)))]
        self.root = self._build_tree(X, y, 0)
        return self
    
    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        probs = np.bincount(y, minlength=len(self.class_names)) / len(y)
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _best_split(self, X, y):
        best_impurity, best_feature, best_threshold = float('inf'), None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                impurity = (np.sum(left_mask) * self._impurity(y[left_mask]) + 
                           np.sum(right_mask) * self._impurity(y[right_mask])) / len(y)
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split):
            return Node(value=np.bincount(y).argmax() if len(y) > 0 else 0)
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax() if len(y) > 0 else 0)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature, threshold, left, right)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        return self._predict_one(x, node.left if x[node.feature] <= node.threshold else node.right)
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}→ Class: {self.class_names[node.value]}")
        else:
            print(f"{indent}If {self.feature_names[node.feature]} <= {node.threshold:.2f}:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Else:")
            self.print_tree(node.right, depth + 1)

def evaluate(tree, X, y, test_data, expected):
    tree.fit(X, y, feature_names=["Alcohol (%)", "Sugar (g/L)", "Color"], class_names=["Wine", "Beer", "Whiskey"])
    print(f"\n{tree.criterion.capitalize()}-based Tree:")
    tree.print_tree()
    preds = tree.predict(test_data)
    predicted_labels = [tree.class_names[p] for p in preds]
    for i, (features, pred, exp) in enumerate(zip(test_data, predicted_labels, expected)):
        print(f"Sample {i+1}: {features} → Predicted: {pred}, Expected: {exp}")
    accuracy = np.mean(predicted_labels == np.array(expected))
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    X, y, class_names = preprocess_data(data)
    test_data = np.array([[6.0, 2.1, 0], [39.0, 0.05, 1], [13.0, 1.3, 1]])
    expected = ["Beer", "Whiskey", "Wine"]
    
    for criterion in ['gini', 'entropy']:
        tree = DecisionTreeClassifier(max_depth=3, criterion=criterion)
        evaluate(tree, X, y, test_data, expected)


    