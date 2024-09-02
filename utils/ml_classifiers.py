from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dt': DecisionTreeClassifier(random_state=42),
    'rf': RandomForestClassifier(n_estimators=10, random_state=42),
    'mlp': make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, learning_rate='invscaling',
                                       random_state=42)),
    'svm': SVC(probability=True, random_state=42),
    # 'svm': SVC(gamma=0.001, C=100., kernel='linear'),
    # 'svm': SVC(kernel='linear', C=1.0, gamma=0.001, random_state=42),
    'lr': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
}
