import keras
from keras.layers import Input, Dense , LSTM,Dropout,TimeDistributed,Bidirectional
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import seaborn as sns
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

#Preprocessing steps performed on the data, with an option of creating validation set
def processData(df,validOption,numberofstages):
    #Input and output columns to LSTM are just the p,d,s and c
    #They are the unit production cost, demand, setup and holding cost
    input_cols=[col for col in df if (col.startswith('p[')| col.startswith('d[')|col.startswith('s[')|col.startswith('c['))]
    #output column is the y variable, which is the production decision
    output_cols=[col for col in df if col.startswith('y[')]
    
    #Cplex can give values close to zero instead of zeroand it is shown by -0.0.
    #Make them either strict zero or one
    df[output_cols] = df[output_cols].apply(lambda row:binarylabeler(row,0.5),axis=0)
    
    #Split data as train,validation(for hyperparameter tuning) and test sets
    #x represents input and y represents the output for the deep learning model
    x = df.loc[:,input_cols]
    y = df.loc[:,output_cols]
    #If validOption is True, create train, validation and test set with split of 0.64, 0.16 and 0.2
    if validOption==True:
        x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
        x_train,x_valid, y_train, y_valid=train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    #If validOption is False, create train and test set with split of 0.8 and 0.2, which is useful for training with more data
    else:
        x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
    
    #output is the dataframe that we write the data with predictions to use in CPLEX solver
    #copy the rows from the df using the indexes if the test set
    output=df.iloc[x_test.index,:].copy()
    
    #This is scaling, considering the validation option
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    if validOption==True:
        x_valid=scaler.transform(x_valid)
    x_test=scaler.transform(x_test)
    #Transform train, validation,and test set to  pandas dataframe and copy the index to prevent a mix up
    x_train=pd.DataFrame(x_train,columns=input_cols)
    x_train.index=y_train.index.copy()
    if validOption==True:
        x_valid=pd.DataFrame(x_valid,columns=input_cols)
        x_valid.index=y_valid.index.copy()
    x_test=pd.DataFrame(x_test,columns=input_cols)
    x_test.index=y_test.index.copy()
    
    
    #create a new dataset with appropriate form for LSTM input, considering the validation option
    #The LSTM input layer must be 3 Dimensional as (samples, time steps, and features)
    #Basically npx_train and x_train are the same but first is 2D and other is 3D etc.
    #First get column names for each feature and output
    feature_p=[col for col in df if col.startswith('p[')]
    feature_d=[col for col in df if col.startswith('d[')]
    feature_c=[col for col in df if col.startswith('c[')]
    feature_s=[col for col in df if col.startswith('s[')]
    output_cols=[col for col in df if col.startswith('y[')]  
        
    #The following loops organizes datasets as proper input to LSTM
    #Training set
    npx_train = np.empty(shape=(len(x_train),numberofstages,int(len(x_train.columns)/numberofstages)))
    npy_train = np.empty(shape=(len(x_train),numberofstages,1))
    for i in range(len(x_train)):
        npx_train[i,:,0]=x_train.loc[x_train.index[i],feature_p]
        npx_train[i,:,1]=x_train.loc[x_train.index[i],feature_d]
        npx_train[i,:,2]=x_train.loc[x_train.index[i],feature_c]
        npx_train[i,:,3]=x_train.loc[x_train.index[i],feature_s]    
        npy_train[i,:,0]=y_train.loc[x_train.index[i],output_cols]
    #Validation set - Optional
    if validOption==True:
        npx_valid = np.empty(shape=(len(x_valid),numberofstages,int(len(x_valid.columns)/numberofstages)))
        npy_valid = np.empty(shape=(len(x_valid),numberofstages,1))
        for i in range(len(x_valid)):
            npx_valid[i,:,0]=x_valid.loc[x_valid.index[i],feature_p]
            npx_valid[i,:,1]=x_valid.loc[x_valid.index[i],feature_d]
            npx_valid[i,:,2]=x_valid.loc[x_valid.index[i],feature_c]
            npx_valid[i,:,3]=x_valid.loc[x_valid.index[i],feature_s]    
            npy_valid[i,:,0]=y_valid.loc[x_valid.index[i],output_cols]      
    npx_test = np.empty(shape=(len(x_test),numberofstages,int(len(x_test.columns)/numberofstages)))
    npy_test = np.empty(shape=(len(x_test),numberofstages,1))
    #Test Set
    for i in range(len(x_test)):
        npx_test[i,:,0]=x_test.loc[x_test.index[i],feature_p]
        npx_test[i,:,1]=x_test.loc[x_test.index[i],feature_d]
        npx_test[i,:,2]=x_test.loc[x_test.index[i],feature_c]
        npx_test[i,:,3]=x_test.loc[x_test.index[i],feature_s]    
        npy_test[i,:,0]=y_test.loc[x_test.index[i],output_cols]
    
    if validOption==True:
        return output,x_train,y_train,x_valid,y_valid,x_test,y_test,npx_train,npy_train,npx_valid,npy_valid,npx_test,npy_test
    else:
        return output,x_train,y_train,x_test,y_test,npx_train,npy_train,npx_test,npy_test

#Train the model given parameters        
def trainModel(npx_train,npy_train,opt,number_of_hidden_units,dropout,numberofstages,batch_size,modelPath):
    #Bidirectional LSTM model 
    #input shape is (numberofstages,number of features)
    inputlayer=Input(shape=(numberofstages,npx_train.shape[2]))
    lstm1=Bidirectional(LSTM(number_of_hidden_units,return_sequences=True,activation='tanh'))(inputlayer)
    dropout1=Dropout(dropout)(lstm1)
    lstm2=Bidirectional(LSTM(number_of_hidden_units,return_sequences=True,activation='tanh'))(dropout1)
    dropout2=Dropout(dropout)(lstm2)
    lstm3=Bidirectional(LSTM(number_of_hidden_units,return_sequences=True,activation='tanh'))(dropout2)
    dropout3=Dropout(dropout)(lstm3)
    outputlayer = TimeDistributed(Dense(1,activation='sigmoid'))(dropout3)
    model = Model(inputs=inputlayer, outputs=outputlayer)
    #Compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    #Stop training if there is no improvement in validation loss for 30 epochs
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    #Save model based on the validation loss
    mc = keras.callbacks.ModelCheckpoint(modelPath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #Terminate if there is some nan error
    nanstop=keras.callbacks.callbacks.TerminateOnNaN()
    #Reduce learning rate by half if there are no improvements for 5 epochs
    reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00000001,cooldown=1)
    # fit the keras model on the dataset
    t0 = time.time()
    #fit=model.fit(npx_train, y_train,epochs=300,batch_size=128,validation_split=0.2,callbacks=[es,mc])
    fit=model.fit(npx_train[:,:,:],npy_train[:,:,],epochs=2000,batch_size=batch_size,validation_split=0.2,callbacks=[es,mc,nanstop,reduce_lr])
    t1 = time.time()
    print('Runtime :',t1-t0)
    return fit

#load the trained model
def loadModel(modelPath):    
    return keras.models.load_model(modelPath)
    
#Predict given model, test set in LSTM format and index set as in regular format
def predictModel(model,test_set,index_set):
    #Calculate prediction time
    t0 = time.time()
    predictions = model.predict(test_set[:,:,:])
    t1 = time.time()
    print('Prediction time:',t1-t0)
    #get the predictions to predictions_df and preserve the index
    predictions_df=index_set.copy()
    for i in range(len(test_set[:,:,:])):
        predictions_df.loc[index_set.index[i],:]=predictions[i,:,0]
    return predictions_df

def saveFigures(fit,figurePath,y_test,predictions):
    #Loss
    #Plot loss
    fig, ax = plt.subplots()
    loss = fit.history['loss']
    val_loss = fit.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ax.plot(epochs, loss, 'g', label='Training loss')
    ax.plot(epochs, val_loss, 'y', label='Validation loss')
    ax.set(title='Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    fig.savefig(figurePath+'-loss.png', dpi=1000)
    #plot learning rate
    fig, ax = plt.subplots()
    lrplot = fit.history['lr']
    epochs = range(1, len(lrplot) + 1)
    ax.plot(epochs, lrplot, 'g', label='Training loss')
    ax.set(title='Learning rate vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    #plt.show()
    fig.savefig(figurePath+'-lr.png', dpi=1000)
    
    #Plot ROC curve
    #First flatten true and predicted values to get the roc curve
    flatteny_test=pd.Series([])
    flattenpredictions=pd.Series([])
    temp=pd.DataFrame(predictions)
    for i in y_test.columns:
        flatteny_test=flatteny_test.append(y_test[i], ignore_index=True)
    for i in temp.columns:
        flattenpredictions=flattenpredictions.append(temp[i], ignore_index=True)
    lr_auc = roc_auc_score(flatteny_test, flattenpredictions)
    # calculate roc curves
    fpr, tpr,_ = roc_curve(flatteny_test, flattenpredictions)
    # plot the roc curve for the model
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, linestyle='--', label='')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC=%.3f' % (lr_auc))
    #plt.show()
    fig.savefig(figurePath+"roc.png", dpi=300)
    
    #Plot Histogram
    plt.hist(flattenpredictions,bins=20)
    plt.title('Histogram of Predicted Probabilities')
    plt.gcf().savefig(figurePath+"hist.png")
    plt.clf()
    #plot heatmap
    heat_map = sns.heatmap(predictions)
    fig = heat_map.get_figure()
    plt.title('Heatmap of Probabilities')
    fig.savefig(figurePath+"heatmap.png", dpi=1000)

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
        #Use try except because to prevent key errors since some labels might not exist
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


#input path
inputPath='/data/resultsfromclspsolvercd8f1000t90n100k.txt'       
#number os stages of the clsp instances
numberofstages=90   
df=readData(inputPath,numberofstages)

#This is the model path that used for saving the model
modelPath='/results/c8f1000t90.h5'

#This is the path that used for saving the model plots such as loss, learning rate, Roc, histogram and heatmap of predictions 
#This is just for visualization and does not change the training 
figurePath='/results/plot'

#Output path to write results
outputPath='/results/c8f1000t90'

#If validation set is used data is divided as: Train 64%, Validation 16%, Test 20%
#If validation set is not used data is divided as: Train 80%, Test 20%
#validationOption=False is used for training after model hyperparameters are decided using validation set
validationOption=False

#LSTM hyperparameters used for training
learning_rate=0.01
number_of_hidden_units=40
dropout=0.3
batch_size=64
adam=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False,clipnorm=1,clipvalue=1)

#Check the number of columns and rows
print(df.shape)




##########################################################
#If there is no validation set involved:
if validationOption==False:
    #Preprocess the data 
    output,x_train,y_train,x_test,y_test,npx_train,npy_train,npx_test,npy_test=processData(df,False,numberofstages)
    
    #Train the model
    fit=trainModel(npx_train,npy_train,adam,number_of_hidden_units,dropout,numberofstages,batch_size,modelPath)
    
    #load model
    model=loadModel(modelPath)
    
    #Generate predictions
    predictions=predictModel(model,npx_test,y_test)
    
    #Generate and save figures
    saveFigures(fit,figurePath,y_test,predictions)
    
    #save predictions with the outout dataframe for different level of predictions
    saveTestSetwithPredictions(output,outputPath,y_test,predictions)



##########################################################
#If validation set option is true
if validationOption==True:
    #Preprocess the data with validation set
    output,x_train,y_train,x_valid,y_valid,x_test,y_test,npx_train,npy_train,npx_valid,npy_valid,npx_test,npy_test=processData(df,True,numberofstages)
    
    #Train the model
    fit=trainModel(npx_train,npy_train,adam,number_of_hidden_units,dropout,numberofstages,batch_size,modelPath)
    
    #load model
    model=loadModel(modelPath)
    #Generate predictions for validation set
    predictions=predictModel(model,npx_valid,y_valid)
    
    #Generate and save figures
    saveFigures(fit,figurePath,y_valid,predictions)
    
    #save predictions with the outout dataframe for different level of predictions
    saveTestSetwithPredictions(output,outputPath,y_valid,predictions)

