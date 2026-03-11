import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Separate input and output
X = data[["Hours"]]
y = data["Score"]

# Train Machine Learning Model
model = LinearRegression()
model.fit(X, y)

# Predict score for new study hours
hours = [[7.5]]
predicted_score = model.predict(hours)

print("Predicted Score for 7.5 study hours:", round(predicted_score[0],2))

# Visualization
plt.scatter(data["Hours"], data["Score"], color="blue")
plt.plot(data["Hours"], model.predict(X), color="red")

plt.title("Study Hours vs Score Prediction")
plt.xlabel("Hours Studied")
plt.ylabel("Score")

plt.savefig("prediction_graph.png")

print("Graph saved as prediction_graph.png")

