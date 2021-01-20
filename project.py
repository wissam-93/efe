# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:40:46 2021

@author: wessam
"""


"""

Columns description

anaemia     :Decrease of red blood cells or hemoglobin (boolean)

creatinine_phosphokinase:   Level of the CPK enzyme in the blood (mcg/L)

diabetes:    If the patient has diabetes (boolean)

ejection_fraction:   Ejection fraction (EF) is a measurement, 
expressed as a percentage, of how much blood the left ventricle pumps out with each contraction

high_blood_pressure:  blood hypertension

platelets   : component of blood whose function (along with the coagulation factors)

serum_creatinine:   Serum creatinine is widely interpreted as a measure only of renal function

serum_sodium: to see how much sodium is in your blood it is particularly important for nerve and muscle function.



Sex - Gender of patient Male = 1, Female =0
Age - Age of patient
Diabetes - 0 = No, 1 = Yes
Anaemia - 0 = No, 1 = Yes
High_blood_pressure - 0 = No, 1 = Yes
Smoking - 0 = No, 1 = Yes
DEATH_EVENT - 0 = No, 1 = Yes

Binary 
"""





import pandas as pd

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')





# Checking our Dataset + Basic Operations

"""
1.

x = pd.DataFrame(df.describe()) # indataframe 
print(x.describe())

The describe() function computes a summary of statistics 
only numeric coulumns

"""



"""
2.

print(x.info())
info() method prints information about a DataFrame 
including the index dtype and column dtypes, non-null values and memory usage
"""


"""
3.
print(x.head())
Head()  method return top 5 rows of a data frame
"""



"""
4.
print(x.dtypes)
#our data is only numurical
dtype Return a subset of the DataFrame’s columns based on the column dtypes.
"""


"""
5.

#checking our columns list
Print(list(x.columns))

"""




# Data  Cleaning
"""
print(x.isnull().sum())

#There is no Missing Values

"""






#Check coorelation between each column "variables"

#x = pd.DataFrame(df.corr())

"""
correlation between Death Event and other features, 
the corrolation between death_event and sex ,diabetes is weak!!

ourtarget is death event
"""




#drop sex+diabetes
df = pd.DataFrame(df.drop('sex', axis=1))
df = pd.DataFrame(df.drop('diabetes', axis=1))



#print(df.shape)


X=df.drop(['DEATH_EVENT'],axis=1)
y=df['DEATH_EVENT']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=5)

"""
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
"""

from sklearn.linear_model import LogisticRegression




X=df[['creatinine_phosphokinase','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','smoking','time']].values
y=df[['DEATH_EVENT']].values
#SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
y_train=y_train.ravel()
y_test=y_test.ravel()






model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


print(model.score(X_test,y_test))




