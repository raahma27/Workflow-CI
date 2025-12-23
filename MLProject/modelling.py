import os
import pandas as pd 
import mlflow 
import mlflow.sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

def main(): 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        BASE_DIR,
        "airline_preprocessed",
        "airline_preprocessed.csv"
    )

    df = pd.read_csv(data_path) 
    # Split fitur & target 
    X = df.drop(columns=["satisfaction"]) 
    y = df["satisfaction"] 
    
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
        ) 
    
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test) 
    
    model = LogisticRegression(max_iter=1000) 
    
    mlflow.sklearn.autolog() 
    
    with mlflow.start_run(): 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        acc = accuracy_score(y_test, y_pred) 
        
if __name__ == "__main__": 
    main()