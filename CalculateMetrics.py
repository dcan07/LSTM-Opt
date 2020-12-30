import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Given objective function values without and with ML, calculate the optgap
def optgap(z, zhat): 
    z, zhat = np.array(z), np.array(zhat)
    return np.mean(np.abs((z - zhat) / z)) * 100

def acc(true,pred):
    true==pred

#Function to label the data based on the give cutoff value
#Label 0 if the value in the array is smaller than or equal to cutoff value, otherwise label 1
def binarylabeler(col,cutoff):
    return pd.Series([0 if x <= cutoff else 1 for x in col])

#Inputs:
#objectiveValues contains the results from the CPLEX resolve with columns ['solutionTimeWithoutML','solutionTimeWithML','objectiveWithoutML','objectiveWithML'], which also can be aggregate results for some number of datasets
#decisionVariableTrue and decisionVariablePred are the true and predicted values for decision variable y, respectively
#calculatedMetrics is the dataframe which calculate metrics are recorded
#rowid is the current row the metrics are written on the calculatedMetrics
#y is the prediction level
#Calculations are made several times for each prediction level, UC, MS, and averages therefore it is more useful to have it as a function
def calculateMetrics(objectiveValues,decisionVariableTrue,decisionVariablePred,calculatedMetrics,rowid,y):
    #Calculate the number of infeasible        
    infeasible=objectiveValues[objectiveValues['objectiveWithML']==np.str_('Infeasible')]
    #Calculate the percent of infeasibility
    calculatedMetrics.loc[rowid,'infeasible(%)']=len(infeasible)*100/len(objectiveValues)
    #Eliminate infeasible instances
    objectiveValues=objectiveValues[objectiveValues['objectiveWithML']!=np.str_('Infeasible')]
    #Make sure the type is int
    objectiveValues[['objectiveWithML']]=objectiveValues[['objectiveWithML']].astype(int)
    #Calculate the optimality gap
    calculatedMetrics.loc[rowid,'optgap(%)']=optgap(objectiveValues[['objectiveWithoutML']],objectiveValues[['objectiveWithML']])
    #Calculate the mean solution time without ML
    calculatedMetrics.loc[rowid,'soltime']=(objectiveValues[['solutionTimeWithoutML']].sum()[0])/len(objectiveValues)
    objectiveValues[['solutionTimeWithML']]=objectiveValues[['solutionTimeWithML']].astype(float)
    #Calculate the mean solution time with ML
    calculatedMetrics.loc[rowid,'soltimeML']=objectiveValues[['solutionTimeWithML']].sum()[0]/len(objectiveValues)
    #Calculate the time gain
    calculatedMetrics.loc[rowid,'timegain(%)']=((objectiveValues[['solutionTimeWithoutML']].sum()[0]-objectiveValues[['solutionTimeWithML']].sum()[0])*100/objectiveValues[['solutionTimeWithoutML']].sum()[0])
    #Calculate the improvement factor
    calculatedMetrics.loc[rowid,'improvement']=((objectiveValues[['solutionTimeWithoutML']].sum()[0])/objectiveValues[['solutionTimeWithML']].sum()[0])
    #Set prediction Level
    calculatedMetrics.loc[rowid,'predicted(%)']=y
       
    
    #Do not consider label -1(indecisive) for accuracy and F1 calculation
    #Use try except to ensure that there is no key error since 100% prediction level will not have indecisive label
    #Known indexes are stored in certain array
    try:
        certain=decisionVariablePred[decisionVariablePred!=-1].index
    except KeyError:
        certain=decisionVariablePred.index  
     
    #Calculate the percent accuracy
    calculatedMetrics.loc[rowid,'accuracy(%)']=accuracy_score(decisionVariableTrue[certain], decisionVariablePred[certain])*100
    #Calculate the F1 score
    calculatedMetrics.loc[rowid,'F1-score']=f1_score(decisionVariableTrue[certain], decisionVariablePred[certain])
    
    #move to next row
    rowid+=1
    
    return calculatedMetrics,rowid

#Function to run the main loops and calculate metrics for all datasets in modelNames
#Function inputs are defined below
def evaluateDatasets(inputPath,outputPath,predictionLevels,otherApproaches,numberOfStagesArray,meanPredictionTime,modelNames):
    #Following dataframe holds the results of calculations and it becomes the outout file in the end
    calculatedMetrics=pd.DataFrame(index=range(len(predictionLevels)**4),columns=['models','soltime',	'predicted(%)','accuracy(%)','F1-score','soltimeML','timegain(%)','improvement','infeasible(%)','optgap(%)'])
    
    #The following dataframes holds the aggregate decision variables, predictions of decision variables and objective values for all data in the modelNames for prediction level 25 to 100
    decisionVariableTrue_Overall=pd.Series([])
    decisionVariablePred_Overall=pd.Series([])
    objectiveValues_Overall=pd.DataFrame()
    #The following dataframes holds the aggregate decision variables, predictions of decision variables and objective values for all data in the modelNames for MS approach
    decisionVariableTrue_Overall_MS=pd.Series([])
    decisionVariablePred_Overall_MS=pd.Series([])
    objectiveValues_Overall_MS=pd.DataFrame()
    #The following dataframes holds the aggregate decision variables, predictions of decision variables and objective values for all data in the modelNames for UC approach
    decisionVariableTrue_Overall_UC=pd.Series([])
    decisionVariablePred_Overall_UC=pd.Series([])
    objectiveValues_Overall_UC=pd.DataFrame()
    #The following dataframes holds the aggregate decision variables, predictions of decision variables and objective values for all data in the modelNames for 85% prediction level
    decisionVariableTrue_Overall_85=pd.Series([])
    decisionVariablePred_Overall_85=pd.Series([])
    objectiveValues_Overall_85=pd.DataFrame()
    
    rowid=0
    #for each model set, calculate the metrics
    for i,x in enumerate(modelNames):
        print(rowid)
        #Print model name
        calculatedMetrics.loc[rowid,'models']=str(x)
        #Get the number of stages
        numberOfStages=numberOfStagesArray[i]
        
        #The following dataframes holds the aggregate decision variables, predictions of decision variables and objective values for a single dataset in the modelNames for prediction level 25 to 100
        #Singe set dataframes holds the data for 25% to 100% level for just a single set. E.g c=3, f=1,000, T=90
        #Overall dataframes holds the data for all sets examined together. E.g c=3,5,8, f=1,000, T=90
        decisionVariableTrue_SingleSet=pd.Series([])
        decisionVariablePred_SingleSet=pd.Series([])
        objectiveValues_SingleSet=pd.DataFrame()
            
        #Calculate the metrics for each prediction levels in the model set
        for j,y in enumerate(predictionLevels):
        
            #Read Data
            objectiveValues = pd.read_csv(inputPath+'results'+x+'-percent'+str(y)+'.txt',  header=None,sep=';',names=['solutionTimeWithoutML','solutionTimeWithML','objectiveWithoutML','objectiveWithML'],index_col=False )
                    
            #Add mean prediction generation time to each instance
            objectiveValues.loc[:,'solutionTimeWithML']=objectiveValues.loc[:,'solutionTimeWithML']+meanPredictionTime[i]
            
            #In order to calculate the accuracy and F1 score we need to read actual and predicted values for decision variable from another file
            predictions=pd.read_csv(inputPath+x+'-percent'+str(y)+'.txt',  header=None,sep=' ',index_col=False )
        
            #Create arrays to hold true and predicted y decision variables
            decisionVariableTrue=pd.Series(predictions.loc[:,((numberOfStages*6)+4):((numberOfStages*7)+3)].stack().values)
            decisionVariablePred=pd.Series(predictions.loc[:,((numberOfStages*8)+5):((numberOfStages*9)+4)].stack().values)
            #Make sure that labels are strictly 0 or 1, because CPLEX can sometime output real close to 0 or 1 but not exactly
            decisionVariableTrue=binarylabeler(decisionVariableTrue,0.5)
            
            #Calculate the metrics
            calculatedMetrics,rowid=calculateMetrics(objectiveValues,decisionVariableTrue,decisionVariablePred,calculatedMetrics,rowid,y)
    
            
            #Following portion concats the dataframes to use in calculations
            #Concatenate current objectiveValues_Overall and objectiveValues to use on average calculations for all datasets in group
            objectiveValues_Overall=pd.concat([objectiveValues_Overall.copy(),objectiveValues.copy()],ignore_index=True)
            #Concatenate current objectiveValues_SingleSet and objectiveValues to use on average calculations for a single dataset
            objectiveValues_SingleSet=pd.concat([objectiveValues_SingleSet.copy(),objectiveValues.copy()],ignore_index=True)
            #Concatenate current decisionVariableTrue_Overall and decisionVariableTrue to use on average calculations for all datasets in group
            decisionVariableTrue_Overall=pd.concat([decisionVariableTrue_Overall.copy(),decisionVariableTrue.copy()],ignore_index=True)
            #Concatenate current decisionVariableTrue_SingleSet and decisionVariableTrue to use on average calculations for a single dataset
            decisionVariableTrue_SingleSet=pd.concat([decisionVariableTrue_SingleSet.copy(),decisionVariableTrue.copy()],ignore_index=True)
            #Concatenate current decisionVariablePred_Overall and decisionVariablePred to use on average calculations for all datasets in group
            decisionVariablePred_Overall=pd.concat([decisionVariablePred_Overall.copy(),decisionVariablePred.copy()],ignore_index=True)
            #Concatenate current decisionVariablePred_SingleSet and decisionVariablePred to use on average calculations for a single dataset
            decisionVariablePred_SingleSet=pd.concat([decisionVariablePred_SingleSet.copy(),decisionVariablePred.copy()],ignore_index=True)
            
            
            #Following portion concats the dataframes to use in calculations only for 85% since we also report the average for it
            if y==85:
                #Concatenate current objectiveValues_Overall_85 and objectiveValues to use on average calculations for all datasets in group
                objectiveValues_Overall_85=pd.concat([objectiveValues_Overall_85.copy(),objectiveValues.copy()],ignore_index=True)
                #Concatenate current decisionVariableTrue_Overall_85 and decisionVariableTrue to use on average calculations for all datasets in group
                decisionVariableTrue_Overall_85=pd.concat([decisionVariableTrue_Overall_85.copy(),decisionVariableTrue.copy()],ignore_index=True)
                #Concatenate current decisionVariablePred_Overall_85 and decisionVariablePred to use on average calculations for all datasets in group
                decisionVariablePred_Overall_85=pd.concat([decisionVariablePred_Overall_85.copy(),decisionVariablePred.copy()],ignore_index=True)
            
        #Calculate the metrics for other approaches such as MS or UC with 100% prediction level
        for j,y in enumerate(otherApproaches):
        
            #Read Data
            objectiveValues = pd.read_csv(inputPath+'results'+x+'-percent100'+y+'.txt',  header=None,sep=';',names=['solutionTimeWithoutML','solutionTimeWithML','objectiveWithoutML','objectiveWithML'],index_col=False )
                    
            #In order to calculate the accuracy and F1 score we need to read actual and predicted values for decision variable from another file for 100%, since UC and MS uses predictions from 100% level
            predictions=pd.read_csv(inputPath+x+'-percent100'+'.txt',  header=None,sep=' ',index_col=False )
            
            #Create arrays to hold true and predicted y decision variables
            decisionVariableTrue=pd.Series(predictions.loc[:,((numberOfStages*6)+4):((numberOfStages*7)+3)].stack().values)
            decisionVariablePred=pd.Series(predictions.loc[:,((numberOfStages*8)+5):((numberOfStages*9)+4)].stack().values)
            #Make sure that labels are strictly 0 or 1, because CPLEX can sometime output real close to 0 or 1 but not exactly
            decisionVariableTrue=binarylabeler(decisionVariableTrue,0.5)
            
            #Calculate the metrics
            calculatedMetrics,rowid=calculateMetrics(objectiveValues,decisionVariableTrue,decisionVariablePred,calculatedMetrics,rowid,y)
    
            #Following portion concats the dataframes to use in calculations only for UC since we also report the average for it
            if y=='uc':
                #Concatenate current objectiveValues_Overall_UC and objectiveValues to use on average calculations for all datasets in group
                objectiveValues_Overall_UC=pd.concat([objectiveValues_Overall_UC.copy(),objectiveValues.copy()],ignore_index=True)
                #Concatenate current decisionVariableTrue_Overall_UC and decisionVariableTrue to use on average calculations for all datasets in group
                decisionVariableTrue_Overall_UC=pd.concat([decisionVariableTrue_Overall_UC.copy(),decisionVariableTrue.copy()],ignore_index=True)
                #Concatenate current decisionVariablePred_Overall_UC and decisionVariablePred to use on average calculations for all datasets in group
                decisionVariablePred_Overall_UC=pd.concat([decisionVariablePred_Overall_UC.copy(),decisionVariablePred.copy()],ignore_index=True)
                    #Following portion concats the dataframes to use in calculations only for UC since we also report the average for it
            if y=='ms':
                #Concatenate current objectiveValues_Overall_MS and objectiveValues to use on average calculations for all datasets in group
                objectiveValues_Overall_MS=pd.concat([objectiveValues_Overall_MS.copy(),objectiveValues.copy()],ignore_index=True)
                #Concatenate current decisionVariableTrue_Overall_MS and decisionVariableTrue to use on average calculations for all datasets in group
                decisionVariableTrue_Overall_MS=pd.concat([decisionVariableTrue_Overall_MS.copy(),decisionVariableTrue.copy()],ignore_index=True)
                #Concatenate current decisionVariablePred_Overall_MS and decisionVariablePred to use on average calculations for all datasets in group
                decisionVariablePred_Overall_MS=pd.concat([decisionVariablePred_Overall_MS.copy(),decisionVariablePred.copy()],ignore_index=True)
        
        #Calcutate the average for prediction levels from 25% to 100% for a single set. E.g. c=3, f=1,0000 and T=90
        #Basically same calculations are done on the aggregate dataframe for a single set
        #Calculate the metrics
        calculatedMetrics,rowid=calculateMetrics(objectiveValues_SingleSet,decisionVariableTrue_SingleSet,decisionVariablePred_SingleSet,calculatedMetrics,rowid,x+'Average')
    
    #Calcutate the average for prediction levels from 25% to 100% for all datasets in the group. E.g. c=3,5,8 f=1,0000 and T=90
    #Calculate the metrics
    calculatedMetrics,rowid=calculateMetrics(objectiveValues_Overall,decisionVariableTrue_Overall,decisionVariablePred_Overall,calculatedMetrics,rowid,'OverallAverage')
     
    #Calcutate the average for prediction levels with MS approach for all datasets in the group. E.g. c=3,5,8 f=1,0000 and T=90
    #Calculate the metrics
    calculatedMetrics,rowid=calculateMetrics(objectiveValues_Overall_MS,decisionVariableTrue_Overall_MS,decisionVariablePred_Overall_MS,calculatedMetrics,rowid,'OverallAverage_MS')
    
    #Calcutate the average for prediction levels with UC approach for all datasets in the group. E.g. c=3,5,8 f=1,0000 and T=90
    #Calculate the metrics
    calculatedMetrics,rowid=calculateMetrics(objectiveValues_Overall_UC,decisionVariableTrue_Overall_UC,decisionVariablePred_Overall_UC,calculatedMetrics,rowid,'OverallAverage_UC')
    
    #Calcutate the average for prediction level 85% for all datasets in the group. E.g. c=3,5,8 f=1,0000 and T=90
    #Calculate the metrics
    calculatedMetrics,rowid=calculateMetrics(objectiveValues_Overall_85,decisionVariableTrue_Overall_85,decisionVariablePred_Overall_85,calculatedMetrics,rowid,'OverallAverage_85')  
    
         
    calculatedMetrics.to_csv(outputPath, header=True, index=None, sep=',')
    
#Input path of the files containing results
inputPath='/results/'   

#Output file to write the results
outputPath='/results/c3f1000results.csv'   
 
#Prediction levels that are used in the experiements
predictionLevels=[25,50,75,85,90,95,100]

#Other approaches that are used in the experiments
#ms represents the MIPStart and UC represents the usercuts approach that are used with 100% prediction level
#They are used to read file and calculate the metric separately from the 25 to 100% prediction level
#for generalization experiments ms is not considered due to its lack of performance
otherApproaches=['ms','uc']

#Number of Stages is an array holds the number of stages for each dataset
numberOfStagesArray=[90,90,90]

#Mean prediction time for a single instance in seconds
meanPredictionTime=[30/20000,30/20000,30/20000]

#Model names contains the names of the results that needs to examined together, which is necessary for calculating averages
#For example, results for f=1,000, T=90, and c=3,5,8 are calculated together
modelNames=['c3f1000t90','c5f1000t90','c8f1000t90']

#Calculate
evaluateDatasets(inputPath,outputPath,predictionLevels,otherApproaches,numberOfStagesArray,meanPredictionTime,modelNames)
