import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

model = None

# Train the model
def train_model(data):
    global model
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"accuracy": accuracy, "f1_score": f1}

# Make a prediction
def make_prediction(input_data):
    global model
    if model is None:
        return {"error": "Model is not trained yet!"}
    
    X_new = pd.DataFrame([input_data])
    prediction = model.predict(X_new)[0]
    confidence = max(model.predict_proba(X_new)[0])

    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}
