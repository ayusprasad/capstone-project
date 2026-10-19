from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_local_classifier_trains_without_registry_access() -> None:
    data = load_wine()
    train_x, test_x, train_y, test_y = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    model.fit(train_x, train_y)
    assert accuracy_score(test_y, model.predict(test_x)) >= 0.8
