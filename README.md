# 🚢 Titanic Survival Prediction — Machine Learning Deployment

This project is part of my **AnalystLab Africa Data Science Internship — Week 7**, focused on **Machine Learning Model Deployment and Real-World Application**.

The project takes a trained Gradient Boosting classification model developed during the internship and makes it available through an interactive **Streamlit web application**.

Users can enter passenger information and receive a prediction of whether the passenger is likely to survive, together with the model's estimated survival probability.

## 🎯 Project Objective

The objective of this project was to take a trained machine learning model and turn it into a usable application that can generate predictions from new user inputs.

The deployment process involved:

* Saving the trained machine learning model
* Saving the feature scaler used during preprocessing
* Building an interactive Streamlit application
* Loading the saved model without retraining
* Accepting user input through a web interface
* Generating predictions and survival probabilities

## 📊 Dataset

The project uses the **Titanic Survival Prediction dataset**.

The target variable is:

* `Survived = 1` — Passenger survived
* `Survived = 0` — Passenger did not survive

The original dataset contains passenger information such as passenger class, age, sex, family information, fare, and port of embarkation.

## 🤖 Machine Learning Model

The deployed model is an **Ensemble Gradient Boosting Classifier**.

The model was trained in the earlier stage of the internship and achieved the following evaluation results:

* **Training Accuracy:** approximately 89.61%
* **Test Accuracy:** approximately 81.01%

The test confusion matrix was:

```text
[[92, 13],
 [21, 53]]
```

This shows that the model was able to correctly classify a substantial proportion of both survivors and non-survivors on the test data.

## 🧮 Features Used by the Model

The deployed application provides the following inputs:

* Passenger Class (`Pclass`)
* Age (`Age`)
* Siblings/Spouses aboard (`SibSp`)
* Parents/Children aboard (`Parch`)
* Fare (`Fare`)
* Sex
* Port of Embarkation

Categorical variables are converted into numerical features using one-hot encoding.

The final model input features are:

```text
Pclass
Age
SibSp
Parch
Fare
Sex_female
Sex_male
Embarked_C
Embarked_Q
Embarked_S
```

## 🔄 Preprocessing

During model development, `Age` and `Fare` were standardized using `StandardScaler`.

The scaler was saved separately so that new user inputs can receive the **same transformation used during model training**.

This is important because a deployed model must receive data in the same format and scale as the data used when it was trained.


## 🌍 Public Deployment

The application was deployed as a public Streamlit web application so users can access and interact with the model through a web browser.

🔗 **Live Application:** Coming soon

The application allows users to enter Titanic passenger information and receive a survival prediction together with the estimated probability of survival.

### Example Output

If the model predicts survival:

> 🎉 Passenger would have SURVIVED!

If the model predicts non-survival:

> ❌ Passenger would NOT have survived.

The application also displays the estimated probability of survival.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Anaconda
* GitHub

## 📁 Project Structure

```text
Titanic-Deployment/
│
├── app.py
├── titanic_gradient_boosting.joblib
├── scaler.joblib
├── README.md
└── requirements.txt
```

## 💾 Saved Model

The trained machine learning model was saved using Joblib:

```python
joblib.dump(model, "titanic_gradient_boosting.joblib")
```

The scaler was also saved:

```python
joblib.dump(scaler, "scaler.joblib")
```

Saving these files allows the application to reuse the trained model without retraining it every time the application starts.

## ▶️ How to Run the Application

### 1. Install the required libraries

Open an Anaconda Prompt or terminal and install the required packages:

```bash
pip install streamlit pandas scikit-learn joblib
```

### 2. Navigate to the project folder

For example:

```bash
cd path/to/Titanic-Deployment
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will start a local development server and provide a local URL similar to:

```text
http://localhost:8503
```

Open the URL in a web browser to access the application.

## 🧪 Testing the Application

The application was tested by entering different combinations of passenger information, including:

* Different passenger classes
* Male and female passengers
* Different ages
* Different family sizes
* Different fares
* Different ports of embarkation

The application successfully returned predictions and corresponding survival probabilities.

## 📌 Important Note About the Predictions

The prediction probability represents the model's estimated probability based on the features supplied by the user.

It should not be interpreted as a historical certainty about whether a particular real passenger survived.

The application is intended as a demonstration of how a trained machine learning model can be converted into an interactive prediction tool.

## 📚 Key Lessons Learned

This project helped me understand that building a machine learning model is only one part of a data science workflow.

I learned how to:

* Save and reload trained machine learning models
* Save preprocessing objects such as scalers
* Build an interactive prediction interface
* Connect a user interface to a machine learning model
* Accept new data and generate predictions
* Test a deployed application
* Organize a machine learning project for real-world use

One of my biggest takeaways from this week is that **a model becomes more useful when people can actually interact with it and use its predictions.**

## 🚀 Future Improvements

Future improvements could include:

* Deploying the application to a public cloud platform
* Adding more comprehensive input validation
* Improving the user interface
* Adding model performance visualizations
* Adding an API endpoint for programmatic predictions
* Monitoring model performance after deployment

## 👩🏽‍💻 Author

**Folashade**

Data Science Intern
**AnalystLab Africa Internship Program**

---

### Week 7 Focus

**Model Deployment & Real-World Application**

This project demonstrates the transition from a trained machine learning model to an interactive application.
