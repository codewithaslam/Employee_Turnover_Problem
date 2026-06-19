import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("employee_turnover.csv")

X = df.drop('Employee_Turnover', axis=1)
y = df['Employee_Turnover']

X_train , X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.1,random_state=42
)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

##Regularization

#L1 Regression
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5,)
lasso.fit(X_train,y_train)


#L2 Regression
ridge = LogisticRegression(penalty='l2', C=1, max_iter=200)
ridge.fit(X_train,y_train)

models = {'Baseline': lr, 'Lasso': lasso, 'Ridge': ridge}

best_acc = 0
best_model = None
for name,model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_model = name
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
print(f"\nBest Model: {best_model}")
print(f"Best Accuracy: {best_acc}")