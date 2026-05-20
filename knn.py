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