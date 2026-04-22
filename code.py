import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import shap
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import warnings
warnings.filterwarnings('ignore')

print("Hybrid Personalized Learning System (Clean Final Version)\n")

#  creating dataset
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "student_id": range(n),
    "prior_knowledge": np.random.beta(3, 2, n),
    "study_hours": np.random.normal(12, 4, n).clip(4, 28),
    "num_interactions": np.random.randint(20, 90, n),
    "avg_quiz_score": np.random.uniform(45, 98, n),
    "sentiment_score": np.random.uniform(-0.8, 0.9, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "age_group": np.random.choice(["18-22", "23-30", "31+"], n),
    "course_type": np.random.choice(["STEM", "Humanities"], n),
})
df.to_csv("student_learning_data.csv", index=False)
print("Dataset saved as student_learning_data.csv")



df["final_performance"] = (
    df["prior_knowledge"] * 0.35 +
    df["avg_quiz_score"] / 100 * 0.30 +
    df["study_hours"] / 25 * 0.20 +
    df["sentiment_score"] * 0.15
) > 0.57

df["final_performance"] = df["final_performance"].astype(int)

print("Dataset shape:", df.shape)

class DKT(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super(DKT, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


X_seq = np.random.rand(n, 10, 5).astype(np.float32)
y_seq = df["final_performance"].values.reshape(-1, 1).astype(np.float32)

X_seq = torch.tensor(X_seq)
y_seq = torch.tensor(y_seq)

dkt_model = DKT()
optimizer = optim.Adam(dkt_model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()


for epoch in range(15):
    optimizer.zero_grad()
    preds = dkt_model(X_seq)
    loss = loss_fn(preds, y_seq)
    loss.backward()
    optimizer.step()




actions = ["Practice", "Video Lesson", "Motivation Content", "Advanced Challenge"]
Q = np.zeros((n, len(actions)))

def rl_recommend(student_id):
    return actions[np.argmax(Q[student_id])]


for _ in range(200):
    for i in range(n):
        reward = 1 if df.loc[i, "final_performance"] == 1 else -1
        action = np.random.randint(len(actions))
        Q[i, action] = 0.9 * Q[i, action] + 0.1 * reward





analyzer = SentimentIntensityAnalyzer()

def generate_content(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score > 0.3:
        return "Positive learner → Provide advanced challenge"
    elif score < -0.3:
        return "Struggling learner → Provide simplified explanation + support"
    else:
        return "Neutral learner → Provide standard quiz"




df_ml = pd.get_dummies(df, columns=["gender", "age_group", "course_type"], drop_first=True)

X = df_ml.drop(["student_id", "final_performance"], axis=1)
y = df_ml["final_performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n Model Accuracy:", round(acc, 4))
print(classification_report(y_test, pred))




explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

print(" SHAP explainability done")



joblib.dump(rf_model, "rf_model.pkl")
torch.save(dkt_model.state_dict(), "dkt_model.pth")

print("Models saved")




def hybrid_system(student_id):
    student = df.iloc[student_id]

    return {
        "RL_Recommendation": rl_recommend(student_id),
        "NLP_Response": generate_content("I am confused with this topic"),
        "DKT_Prediction": float(dkt_model(X_seq[student_id:student_id+1]).detach().numpy()),
        "Profile": {
            "prior_knowledge": float(student["prior_knowledge"]),
            "quiz_score": float(student["avg_quiz_score"]),
            "sentiment": float(student["sentiment_score"])
        }
    }

print("\n_____ FINAL OUTPUT _____")
print(hybrid_system(0))

