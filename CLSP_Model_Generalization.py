import keras
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#Calculates the mean absolute percentage error
def mape(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    return np.mean(np.abs((y - yhat) / y)) * 100

#Function to label the data based on the give cutoff value
#Label 0 if the value in the array is smaller than or equal to cutoff value, otherwise label 1
def binarylabeler(col,cutoff):
    return [0 if x <= cutoff else 1 for x in col]

#Read the data given input path and the number of stages of the dataset
def readData(path, numberofstages):
    
    #Following arrays and loops are just for column naming
    p=[None]*numberofstages
    d=[None]*numberofstages
    c=[None]*numberofstages
    h=[None]*numberofstages
    s=[None]*numberofstages
    x=[None]*numberofstages
    y=[None]*numberofstages
    inv=[None]*numberofstages
    obj=[None]*numberofstages        
    for i in range(numberofstages):
        p[i]='p['+str(i+1)+']'
        d[i]='d['+str(i+1)+']'
        c[i]='c['+str(i+1)+']'
        h[i]='h['+str(i+1)+']'
        s[i]='s['+str(i+1)+']'
        x[i]='x['+str(i+1)+']'
        y[i]='y['+str(i+1)+']'
        inv[i]='inv['+str(i+1)+']'
        obj[i]='obj['+str(i+1)+']'
        
    #names holds the column names for the dataset.
    #it is in the correct order as written from CLSP solver
    names=['capdem']+['sethold']+p+d+c+h+s+['solutiontime']+['objvalue']+x+y+['initialinv']+inv
    df=pd.read_csv(path,  header=None,sep=';',names=names,index_col=False )
    #filter if there are infeasible CLSP problems and recheck the number of instances
    #There will be value error if there are no infeasible instances, prevent it by a try catch
    try:
        df=df[df['objvalue']!='Infeasible']
    except:
        print('No infeasible instances')
    print(df.shape)
    
    return df

#Preprocessing steps performed on the data. No training takes place since we use already trained models to generate predictions
def processData(df,numberofstages):
    #Input and output columns to LSTM are just the p,d,s and c
    #They are the unit production cost, demand, setup and holding cost
    input_cols=[col for col in df if (col.startswith('p[')| col.startswith('d[')|col.startswith('s[')|col.startswith('c['))]
    #output column is the y variable, which is the production decision
    output_cols=[col for col in df if col.startswith('y[')]
    
    #Cplex can give values close to zero instead of zeroand it is shown by -0.0.
    #Make them either strict zero or one
    df[output_cols] = df[output_cols].apply(lambda row:binarylabeler(row,0.5),axis=0)
    
    #Thre is ony the test set
    #x represents input and y represents the output for the deep learning model
    x_test = df.loc[:,input_cols]
    y_test = df.loc[:,output_cols]
 
    #output is the dataframe that we write the data with predictions to use in CPLEX solver
    #copy the rows from the df using the indexes if the test set
    output=df.iloc[x_test.index,:].copy()
    
    #This is scaling
    scaler = preprocessing.StandardScaler().fit(x_test)
    x_test=scaler.transform(x_test)
    
    #Transform test set to  pandas dataframe and copy the index to prevent a mix up
    x_test=pd.DataFrame(x_test,columns=input_cols)
    x_test.index=y_test.index.copy()
    
    
    #create a new dataset with appropriate form for LSTM input, considering the validation option
    #The LSTM input layer must be 3 Dimensional as (samples, time steps, and features)
    #Basically npx_test and x_test are the same but first is 2D and other is 3D
    #First get column names for each feature and output
    feature_p=[col for col in df if col.startswith('p[')]
    feature_d=[col for col in df if col.startswith('d[')]
    feature_c=[col for col in df if col.startswith('c[')]
    feature_s=[col for col in df if col.startswith('s[')]
    output_cols=[col for col in df if col.startswith('y[')]  
             
    npx_test = np.empty(shape=(len(x_test),numberofstages,int(len(x_test.columns)/numberofstages)))
    npy_test = np.empty(shape=(len(x_test),numberofstages,1))
    #Test Set
    for i in range(len(x_test)):
        npx_test[i,:,0]=x_test.loc[x_test.index[i],feature_p]
        npx_test[i,:,1]=x_test.loc[x_test.index[i],feature_d]
        npx_test[i,:,2]=x_test.loc[x_test.index[i],feature_c]
        npx_test[i,:,3]=x_test.loc[x_test.index[i],feature_s]    
        npy_test[i,:,0]=y_test.loc[x_test.index[i],output_cols]
    return output,x_test,y_test,npx_test,npy_test


#load the trained model
def loadModel(modelPath):    
    return keras.models.load_model(modelPath)
    
#Predict given model, test set in LSTM format and index set as in regular format
def predictModel(model,test_set,index_set,numberofstagesinData,numberofstagesinModel):
    #Calculate prediction time
    t0 = time.time() 
    #If numberofstagesinData=360 and numberofstagesinModel=90
    #multiple is 4, meaning we must generate 4 set of predictions
    multiple=int(numberofstagesinData/numberofstagesinModel)
    #get the predictions to predictions_df and index is preserved since we just have test set
    predictions_df=index_set.copy()
    #for example if numberofstagesinData=360 and numberofstagesinModel=90
    #We must generate 4 set of predictions 1..90,91..180,181..270,271..360 and combine them 
    #for each multiple of the test set, generate and set the predictions
    for j in range(multiple):
        predictions_df.iloc[:,j*numberofstagesinModel:(j+1)*numberofstagesinModel]=model.predict(test_set[:,j*numberofstagesinModel:(j+1)*numberofstagesinModel,:]).squeeze()
    t1 = time.time()
    print('Prediction time:',t1-t0)
    return predictions_df

#Calculate the lbels as 0,1 or -1(indecisive) based on the prediction level given
def partitionPredictions(row,percentpred):
    #Get the maximum of (a,1-a)
    pred=[1-a if 1-a>a else a for a in row]
    #calculate the number of predictions needed based on percentage
    numberpred=int(round(len(pred)*percentpred/100))
    #get the index of highest values
    ind = pd.Series(np.argpartition(np.array(pred), -numberpred)[-numberpred:])
    #for each point in instance
    for a in range(len(row)):
        #if it is in included in ind (meaning one of the max(x,1-x) at the desired level)
        if a in ind.values:
            #label 0 if smaller than or equal to 0.5
            if row[a]<=0.5:
                row[a]=0
            #otherwise label 1
            else:
                row[a]=1
        #if not in the highest values indexes, than label -1 as indecisive 
        else:
            row[a]=-1
    return row

#Calculate labels using with different levels and save in the proper format
def saveTestSetwithPredictions(output,outputPath,y_test,predictions):
    #Consider prediction level from 25% to 100% as below
    predictionLevel=[25,50,75,85,90,95,100]
    #overallResults holds overall profile of predictions
    overallResults=pd.DataFrame(index=range(len(predictionLevel)**2),columns=['percentlabel0','percentlabel1','percentinconclusive','percent predicted','percent accuracy'])
    count=0
    #for each prediction level
    for j,k in enumerate(predictionLevel):
        #Get the labels
        binarypredictions= pd.DataFrame(predictions).copy().apply(lambda row:partitionPredictions(row,k),axis=1)
        #Flatten both actual and predicted to calculate some measures
        flatteny_test=pd.Series([])
        flattenpredictions=pd.Series([])
        temp=pd.DataFrame(binarypredictions)
        for i in y_test.columns:
            flatteny_test=flatteny_test.append(y_test[i], ignore_index=True)
        for i in temp.columns:
            flattenpredictions=flattenpredictions.append(temp[i], ignore_index=True)
        #calculate the proportions of label 0,1 and -1(indecisive)
        #Use try except because to prevent key errors
        try:
            overallResults.loc[count,'percentlabel0']=flattenpredictions.value_counts()[0]/len(flatteny_test)*100
        except KeyError:
            overallResults.loc[count,'percentlabel0']=0
        try:
            overallResults.loc[count,'percentlabel1']=flattenpredictions.value_counts()[1]/len(flatteny_test)*100
        except KeyError:
            overallResults.loc[count,'percentlabel1']=0
        try:
            overallResults.loc[count,'percentinconclusive']=flattenpredictions.value_counts()[-1]/len(flatteny_test)*100
        except KeyError:
            overallResults.loc[count,'percentinconclusive']=0
        #Get the percent predicted
        overallResults.loc[count,'percent predicted']=k
        #calculate Accuracy by only using label 0 and 1
        #do not consider label -1 because it means indecisiv
        try:
            certain=flattenpredictions[flattenpredictions!=-1].index
        except KeyError:
            certain=flattenpredictions.index
        overallResults.loc[count,'percent accuracy']=accuracy_score(flatteny_test[certain], flattenpredictions[certain])*100
        #write file
        temp=pd.DataFrame(binarypredictions)
        output.reset_index(drop=True, inplace=True)
        temp.reset_index(drop=True, inplace=True)
        temp1=pd.concat([output,temp],axis=1)
        #temp1.drop(features,axis=1,inplace=True)
        temp1.to_csv(outputPath+'-'+'percent'+str(k)+'.txt', header=None, index=None, sep=' ')
        count+=1
    
    overallResults.dropna(inplace=True,how='all' )
    overallResults.to_csv(outputPath+'overallResults.csv', header=True, index=None, sep=',')


#input path to read the data
inputPath='/data/resultsfromclspsolvercd3f1000t180n20k.txt'       
#number os stages of the clsp instances, and the trained LSTM model
numberofstagesinData=180
numberofstagesinModel=90
df=readData(inputPath,numberofstagesinData)

#This is the model path that used for loading the trained model
modelPath='/results/c3f1000t90.h5'

#Output path to write results
outputPath='/results/c3f1000t90generalization'

#Check the number of columns and rows
print(df.shape)

#Preprocess the data depending on the option of using a validation set or not
#After optimal hyperparameters are found, no need for validation for training since it will used less data
output,x_test,y_test,npx_test,npy_test=processData(df,numberofstagesinData)

#load model
model=loadModel(modelPath)

#Generate predictions
predictions=predictModel(model,npx_test,y_test,numberofstagesinData,numberofstagesinModel)

#save predictions with the outout dataframe for different level of predictions
saveTestSetwithPredictions(output,outputPath,y_test,predictions)

