class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, a, b):
        total = 0
        for index in range(len(a)):
            total = total + (a[index] - b[index]) ** 2
        return total ** 0.5
    
    def predict(self, X):
        predictions = []
        for row in X:
            distances = []
            for point in self.X_train:
                d = self._distance(point, row)
                distances.append(d)
            k_plus_proches = sorted(zip(distances, self.y_train), key=lambda x: x[0])[:self.k]
            labels = [label for (distance, label) in k_plus_proches]
            predictions.append(max(labels, key=labels.count))
        return predictions
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return sum(1 for p, vrai in zip(predictions, y) if p == vrai)/len(y)