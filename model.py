import pandas as pd
import pickle

#Reading dataset
dataset = pd.read_csv('updated_db.csv')
dataset.head(3)

#Mapping strings to integer values
dataset['FIT']=dataset['FIT'].map({'YES': 1, 'NO': 0})
X=dataset.iloc[:,[0,1,2]].values
y=dataset.iloc[:,4].values

#Splitting dataset into training and testing sets 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Pre-processing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Fitting
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto')
classifier.fit(x_train,y_train)

print("Scaler mean: ", sc.mean_)
print("Scaler scale: ", sc.scale_)

#Dumping the ML model using pickle
pickle.dump(classifier,open('model.pkl','wb'))

