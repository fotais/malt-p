import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn import svm
#from sklearn.inspection import DecisionBoundaryDisplay
import os
import requests
from pathlib import Path  
from sklearn.ensemble import RandomForestClassifier
import time
import csv
import subprocess
import pickle
from csv import writer

###This global dictionary will save the model profiles for each type of RW
psm_profiles = {'small': None, 'medium': None, 'heavy': None}
simulatorCalls=0


###Calls the actual Simulink test script for a test case
def callRwSimulator(torque, duration, interval, size):
        #copies specification of the simulation based on size
        #subprocess.run(['cp', '-f', 'DSF/database/DSF_SDB_'+size+'.xls', 'DSF/database/DSF_SDB.xls'])		
 
        phase2_torque = torque
        phase2_duration = duration
        phase3_interval = interval
 
        # creates subdirectory to store test results of current run
        localJobPath = 'DSF/test/unit/' +'TEST-' + size + '-' +str(phase2_torque)+'_'+str(phase2_duration)+'_'+str(phase3_interval)
        subprocess.run(['mkdir', '-p', localJobPath])

        subprocess.run(['echo', '---------- STARTING ITERATION on ' + size + ' ----------'])
        subprocess.run(['echo', 'TORQUE: ', str(phase2_torque)])
        subprocess.run(['echo', 'DURATION: ', str(phase2_duration)])
        subprocess.run(['echo', 'INTERVAL: ', str(phase3_interval)])
          
        #### SIMULATION ####
        # makes the run-simulation.sh script executable
        subprocess.run(['chmod', '+x','code/UC1-RW/run-simulation.sh'])

        # runs the matlab simulation synchronously
        subprocess.run([
            './code/UC1-RW/run-simulation.sh',
            str(phase2_torque),
            str(phase2_duration),
            str(phase3_interval),
            str(size)
            ])

        subprocess.run(['echo', '### STARTING POST-RUN ###'])
        #subprocess.run(['ls', '-la', 'DSF/test/unit/UT-RWS-002'])

        # copies artefacts to correct test output folder to avoid overwrites after each run
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/sat1_sdb.m', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/sim_sdb.m', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/workspace.mat', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/diary', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/directory.txt', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/test_input.csv', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/test_output.csv', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/rws-passive-run-down-commanded-torque.jpg', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/rws-passive-run-down-torque.jpg', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/rws-passive-run-down-speed.jpg', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/report/test-report.tex', localJobPath])
        subprocess.run(['cp', 'DSF/test/unit/UT-RWS-002/report/test-report.pdf', localJobPath])

        #subprocess.run(['echo', 'CONTENTS OF: DSF/test/unit/UT-RWS-002/'])
        #subprocess.run(['ls', '-la', 'DSF/test/unit/UT-RWS-002'])
        #subprocess.run(['echo', 'CONTENTS OF: ', localJobPath])
        #subprocess.run(['ls', '-la', localJobPath])

        #### TEST RESULT EVALUATION ####
        subprocess.run(['echo', '### STARTING TEST RESULT EVALUATION ###'])
        adequate=evaluateTest(localJobPath, size, phase2_torque, phase2_duration, phase3_interval)
        return adequate
        

def saveTestResultsToFile(profile, inertia, MAX_THRES_SPEED, MAX_TORQUE, torque, duration, interval, max_velocity, time_to_stop, test_adequate, explanation):
    List = [profile, inertia, MAX_THRES_SPEED, MAX_TORQUE, torque, duration, interval, max_velocity, time_to_stop, test_adequate, explanation]
    with open('DSF/all-profile-test-results.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()

def createResultsFile():
    testCases = pd.DataFrame(data=None,  columns=['rw_profile', 'inertia', 'MAX_THRES_SPEED', 'MAX_THRES_TORQUE', 'phase2_torque','phase2_duration', 'phase3_interval', 'max_velocity', 'time_to_stop', 'test_adequate', 'explanation'])
    filepath = Path('DSF/all-profile-test-results.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)
    testCases.to_csv(filepath, index=False)



###PRINT+SEND some Adequate test cases
def generateRandomTests(clf, tests, X_pool, scaling_factor):
    printed=0
    while (printed<tests):
        randIndex = random.randint(0, len(X_pool)-1)
        randTor=X_pool.iloc[randIndex]['phase2_torque']
        randDur=X_pool.iloc[randIndex]['phase2_duration']
        randInt=X_pool.iloc[randIndex]['phase3_interval']
        randCase= pd.DataFrame({"phase2_torque": [randTor],"phase2_duration": [randDur],"phase3_interval": [randInt]})
        if clf.predict(randCase)==1:
            printed=printed+1
            #print('Sending input test case ', printed)
            #plotTestInput(randTor,randDur, randInt)
            numDigits=len(str(int(scaling_factor)))
            test={'phase2_torque': round(randTor/float(scaling_factor),numDigits), 'phase2_duration': int(randDur*100), 'phase3_interval': int(randInt*1000)}
            subprocess.run(['echo', 'Generating test for RW with torque: ', str(round(randTor/float(scaling_factor),numDigits)), ' duration: ', str(int(randDur*100)), ' interval: ', str(int(randInt*1000))])
            
            
def generateAdequateTest(clf, X_pool, scaling_factor):
    while (1):    
        randIndex = random.randint(0, len(X_pool)-1)
        randTor=X_pool.iloc[randIndex]['phase2_torque']
        randDur=X_pool.iloc[randIndex]['phase2_duration']
        randInt=X_pool.iloc[randIndex]['phase3_interval']
        randCase= pd.DataFrame({"phase2_torque": [randTor],"phase2_duration": [randDur],"phase3_interval": [randInt]})
        if clf.predict(randCase)==1:
            numDigits=len(str(int(scaling_factor)))
            test={'phase2_torque': round(randTor/float(scaling_factor),numDigits), 'phase2_duration': int(randDur*100), 'phase3_interval': int(randInt*1000)}
            subprocess.run(['echo', 'Generating test for RW with torque: ', str(round(randTor/float(scaling_factor),numDigits)), ' duration: ', str(int(randDur*100)), ' interval: ', str(int(randInt*1000))])
            return test

def find_most_ambiguous(clf, unknown_indexes, X_pool):
    arr=np.round(list(clf.predict_proba(X_pool.iloc[unknown_indexes]) ), 1)
    
    l=np.where(arr == 0.5)
    if len(l[0])==0:
        l=np.where(arr == 0.4)
#        print("Not l")
    if len(l[0])==0:
        l=np.where(arr == 0.3)
#        print("Not l")
    if len(l[0])==0:
        l=np.where(arr == 0.2)
#        print("Not l")      
    if len(l[0])==0:
        l=np.where(arr == 0.1)
#        print("Not l")
    if len(l[0])==0:
        l=np.where(arr == 1.0)
    ind=l[0][0]    
    return unknown_indexes[ind]


def getValueFromSpecs(spec_file, param):
    dfs = pd.read_excel(spec_file, sheet_name="Constants") 
    if param=='MAX_THRES_SPEED':
        return dfs.iloc[30]['NOM']
    if param=='GAIN':
        return dfs.iloc[14]['NOM']
    if param=='OFFSET':
        return dfs.iloc[43]['NOM']
    if param=='MAX_TORQUE':
        return dfs.iloc[32]['NOM']
    if param=='INERTIA':
        return dfs.iloc[20]['NOM']

def evaluateTest(path, size, torque, duration, interval):
    spec_file_path ="DSF/database/"
    spec_file_name = "DSF_SDB_"+size+".xls"
    output_file_path =path + "/"
 
    inertia = getValueFromSpecs(spec_file_path+spec_file_name, 'INERTIA')
    MAX_THRES_SPEED = getValueFromSpecs(spec_file_path+spec_file_name, 'MAX_THRES_SPEED')
    MAX_TORQUE = getValueFromSpecs(spec_file_path+spec_file_name, 'MAX_TORQUE')

    data = []
    with open(output_file_path+'test_output.csv') as file:
        for line in file:
            line = line.strip()
            data.append(line.split())

    ###added this just to verify the outputs of each test case
    #subprocess.run(['cat', 'DSF/test/unit/UT-RWS-002/init_test.m'])
    #subprocess.run(['cat', output_file_path+'test_output.csv'])
    
    
    df = pd.DataFrame(data)
    x=df.index.values.tolist() 
    y=df.iloc[:, 1]
    y=y.astype(dtype=float)
    #plot(x,y, 'Time', 'Velocity')
    #print('=====================================')
    issue=''
    adequate=1
    max_vel=0
    peak_index=0
    stop_index=0

    ###Identify timepoint for maximum speed
    for index, vel in y.items():
        if vel<max_vel:  #check if velocity just decreased at this timepoint
            #print('maximum velocity= ', max_vel, 'at time: ', peak_index)
            break
        else:
            max_vel=vel
            peak_index=index

    subprocess.run(['echo', 'MAX_VEL: ', str(max_vel)])
    subprocess.run(['echo', 'MAX_THRESH_SPEED: ', str(MAX_THRES_SPEED)])

    if max_vel>MAX_THRES_SPEED*0.95: # was >= 0.9
        adequate=0
        issue='Velocity reached maximum value!'
    elif max_vel<MAX_THRES_SPEED*0.7:  # was 0.4
        adequate=0
        issue='Velocity did not reach a high value, before passive rundown!'

    ###Identify stop timepoint AFTER maximum speed
    for index, vel in y.items():
        if (index>peak_index):
            stop_index=index
            if vel==0:
                subprocess.run(['echo', 'stopped at timepoint= ', str(stop_index)])
                break
    tts=stop_index-peak_index
    subprocess.run(['echo', 'Time to stop = ', str(tts), 's'])

    ###Now check if RW stays stopped AFTER first stop point - a negative spin must follow!!
    if (adequate>0):
        adequate=0
        issue=issue+' Time interval for passive rundown not enough, before negative torque is applied!'    
        for index, vel in y.items():
            if (index<stop_index):
                if vel<0:
                     break #not adequate because the RW did not stay still before neg torque
            if (index==stop_index+2):
                if vel!=0:
                     break #not adequate because the RW did not stay still
            if (index>stop_index+1):
                if vel<0:
                     adequate=1
                     break #adequate because there was negative spin after stop time

    if adequate>0:
        issue=''
    saveTestResultsToFile(size, inertia, MAX_THRES_SPEED, MAX_TORQUE, torque, duration, interval, max_vel, tts, adequate, issue)

    ###Print out the result (test efficacy)    
    if adequate>0:
        subprocess.run(['echo', 'Adequate test case, based on output RW velocity curve.'])
        subprocess.run(['echo', '====================================='])
        gain = getValueFromSpecs(spec_file_path+"DSF_SDB_"+size+".xls", 'GAIN')
        gain=f'{gain:.9f}'
        offset = getValueFromSpecs(spec_file_path+"DSF_SDB_"+size+".xls", 'OFFSET')
        offset=f'{offset:.5f}'

        subprocess.run(['echo', 'DSF_SDB_'+size+'.xls SPEC values, offset: ', offset])
        subprocess.run(['echo', 'DSF_SDB_'+size+'.xls SPEC values, gain: ', gain])
        calcFrictionValues(df,peak_index, stop_index, inertia, max_vel, y, MAX_TORQUE)
    else:
        subprocess.run(['echo', 'Test case not adequate: ', issue])


    #print('=====================================')
    subprocess.run(['echo', '====================================='])

    subprocess.run(['echo', 'Saving feedback...'])
    return adequate

###Replace the torque values depending on the max torque of the specific RW model
def replaceTorqueValues(X, maxT):

        X = X.replace(0.14, maxT)
        X = X.replace(0.13, 13*maxT/14.0)
        X = X.replace(0.12, 12*maxT/14.0)
        X = X.replace(0.11, 11*maxT/14.0)
        X = X.replace(0.1, 10*maxT/14.0)
        X = X.replace(0.09, 9*maxT/14.0)
        X = X.replace(0.08, 8*maxT/14.0)
        X = X.replace(0.07, 7*maxT/14.0)
        X = X.replace(0.06, 6*maxT/14.0)
        X = X.replace(0.05, 5*maxT/14.0)
        X = X.replace(0.04, 4*maxT/14.0)
        X = X.replace(0.03, 3*maxT/14.0)
        X = X.replace(0.02, 2*maxT/14.0)
        X = X.replace(0.01, maxT/14.0)
        return X

### Estimate GAIN and OFFSET from outRWS_Torque:
def calcFrictionValues (df,peak_index, stop_index, inertia, max_vel, yVel, MAX_TORQUE):

        y_tor=df.iloc[:, 0]
        y_tor=y_tor.astype(dtype=float)

        max_tor=0
        min_tor=0
        min_index=0
        est_gain=0
        est_offset=0
        ###Calculate torque slope using dT/dV
        prev_tor=0.0
        prev_index=0
        A=0.0
        B=0.0
        sumA = 0.0
        sumB = 0.0
        yVel=df.iloc[:, 1]
        yVel=yVel.astype(dtype=float)
        ya = y_tor
        for index, tor in y_tor.items():
            if index<=peak_index or index>=stop_index:
                ya[index]=0.0
                prev_tor=tor
                continue
            dTor=tor-prev_tor
            dt=index-prev_index
            if dt==0:
                dt=1

            dVel=yVel[index]-yVel[prev_index]
            if dVel!=0.0:
                A=dTor/dVel
            ya[index]=A
            A=round(A,9)
            sumA=sumA+A
            B=abs(tor)-abs(A)*yVel[index]
            B=round(B,5)
            sumB=sumB+B

            prev_tor=tor
            prev_index=index

        est_gain=abs(sumA/(stop_index-peak_index-1))
        est_gain=f'{est_gain:.9f}'
        est_offset=abs(sumB/(stop_index-peak_index-1))
        est_offset=f'{est_offset:.5f}'
        #print('Estimated values, offset:', est_offset)
        #print('Estimated values, gain:', est_gain)
        
        subprocess.run(['echo', 'Estimated STATIC_FRICTION:', est_offset])
        subprocess.run(['echo', 'Estimated DYNAMIC_FRICTION:', est_gain])

        ###############LAST ADDITION###############
        #####Try to estimate values ONLY with the speed curve (for real RW)
        prev_index=0
        prev_vel=max_vel
        yDec=yVel
        ######First calculate decceleration curve yDec=OutTorque=I*Dv/dt 
        for index, vel in yVel.items():
            dt=index-prev_index
            if dt==0:
                dt=1
                yDec[index]=0.0
            else:
                yDec[index]=inertia*(vel-prev_vel)/dt        
            if abs(yDec[index])>MAX_TORQUE:  #try to ignore deltas
                yDec[index]=0.0
        
            prev_index=index
            prev_vel=vel

            
        ######Calculate Gain=dDec/dV, offset=yDec-V*Gain
        prev_tor=0.0
        prev_index=0
        A=0.0
        B=0.0
        sumA = 0.0
        sumB = 0.0
        yVel=df.iloc[:, 1]
        yVel=yVel.astype(dtype=float)
        ya = yDec
        for index, tor in yDec.items():
            if index<=peak_index+3 or index>=stop_index:
                ya[index]=0.0
                prev_tor=tor
                prev_index=index
                continue
            dTor=tor-prev_tor
            dt=index-prev_index
            if dt==0:
                dt=1

            dVel=yVel[index]-yVel[prev_index]
            A=dTor/dVel
            A=round(A,9)
            sumA=sumA+A
            B=abs(tor)-abs(A)*yVel[index]
            B=round(B,5)
            sumB=sumB+B
            ya[index]=A

            prev_tor=tor
            prev_index=index

        est_gain=abs(sumA/(stop_index-peak_index-4))
        est_gain=f'{est_gain:.9f}'
        est_offset=abs(sumB/(stop_index-peak_index-4))
        est_offset=f'{est_offset:.5f}'
        subprocess.run(['echo', 'Using only output speed & inertia to estimate STATIC_FRICTION:', str(est_offset)])
        subprocess.run(['echo', 'Using only output speed & inertia to estimate DYNAMIC_FRICTION:', str(est_gain)])

        ###############ENDS###############


def saveTestsToFile(X_train):
    run_testCases = pd.DataFrame(data=X_train,  columns=['phase2_torque','phase2_duration', 'phase3_interval', 'result'])
    run_testCases['phase2_torque'] = run_testCases['phase2_torque']*0.1
    run_testCases['phase2_duration'] = run_testCases['phase2_duration']*1000
    run_testCases = run_testCases.astype({"phase2_duration":"int"})
    run_testCases['phase3_interval'] = run_testCases['phase3_interval']*1000
    run_testCases = run_testCases.astype({"phase3_interval":"int"})
    filepath = Path('testcases-to-run.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    run_testCases.to_csv(filepath, index=False)

        
def trainModel(X,y, scaling_factor, profile):
        global simulatorCalls
        ###split the dataset into two parts — pool(80%) and test(20%).
        X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
        X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(drop=True), y_test.reset_index(drop=True)

        ###We take the first 60 indexes/data points of the pool as the initial train data and the rest 480 points as the unlabelled samples
        #train_indexes = list(range(8))
        #unknown_indexes = list(range(8,56))
        train_indexes = list(range(60))
        unknown_indexes = list(range(60, 480))
        X_train = X_pool.iloc[train_indexes]

        ###CALL REAL SIMULATOR: Send the above test cases
        for i, row in X_train.iterrows():
            numDigits=len(str(int(scaling_factor)))
            ##for local tests:
            ##y_pool.iloc[i]=random.choice([0,1])
            ##print('scale with scaling_factor: '+str(scaling_factor), 'value: ', str(row['phase2_torque']))
            y_pool.iloc[i]=callRwSimulator(round(row['phase2_torque']/float(scaling_factor),numDigits), int(row['phase2_duration']*100), int(row['phase3_interval']*1000), profile)
            simulatorCalls=simulatorCalls+1
        y_train = y_pool.iloc[train_indexes]
        
        if len(np.unique(y_train))==1:
            subprocess.run(['echo', 'Problem in Simulator! ONLY 1 CLASS detected for psm profile: '+profile])
            return None
        
        #Create a Kernel Classifier
        model = svm.SVC(kernel="rbf", gamma=10, probability=True).fit(X_train, y_train)

        n = find_most_ambiguous(model, unknown_indexes, X_pool)
        unknown_indexes.remove(n)

        ###we run the active learning algorithm for 180 iterations.
        ###In each of them, we add the most ambiguous {torque,duration,interval} point to the training data and train an SVM, find the most unambiguous point at this stage and then create a plot all this.
        num = 120
        #num = 180
        for i in range(num):

            ###CALL REAL SIMULATOR: get true label for the most ambiguous case from the simulator
            numDigits=len(str(int(scaling_factor)))
            y_pool[n]=callRwSimulator(round(X_pool.iloc[n]['phase2_torque']/float(scaling_factor),numDigits), int(X_pool.iloc[n]['phase2_duration']*100), int(X_pool.iloc[n]['phase3_interval']*1000), profile)
            simulatorCalls=simulatorCalls+1
            ##for local tests:
            ##print(round(X_pool.iloc[n]['phase2_torque']/float(scaling_factor),numDigits), int(X_pool.iloc[n]['phase2_duration']*100), int(X_pool.iloc[n]['phase3_interval']*1000))
            ##y_pool[n]=random.choice([0,1]) 

            ###Add most ambiguous case in training indexes and re-train...
            train_indexes.append(n)
            X_train = X_pool.iloc[train_indexes]
            y_train = y_pool.iloc[train_indexes]
                 
            model = svm.SVC(kernel="rbf", gamma=10, probability=True).fit(X_train, y_train)

            #if i%10==0:
            #    print(' plot_training_data_with_decision_boundary("rbf", X_train , y_train )')
            #    plot_training_data_with_decision_boundary("rbf", X_train[['phase2_torque','phase2_duration']].to_numpy() , y_train.to_numpy() )
            
            n = find_most_ambiguous(model, unknown_indexes, X_pool)
            unknown_indexes.remove(n)

        return model


def trainOnPSMsize(profile):
        global psm_profiles
        spec_file_path ="DSF/database/"
        
        MAX_TORQUE = getValueFromSpecs(spec_file_path+"DSF_SDB_"+profile+".xls", 'MAX_TORQUE')
        
        data = pd.read_csv('code/UC1-RW/all-potential-uc1-inputs-3params.csv', nrows=2744)
        X2 = data[['phase2_torque','phase2_duration','phase3_interval']]
        y2 = data['label']

        #####Replace the torque values depending on the max torque of the specific RW model
        X2=replaceTorqueValues(X2, MAX_TORQUE)
        #####Scale the data in the same climax and discover target boundaries
        X2.loc[:,'phase2_duration']=X2.loc[:,'phase2_duration']/100
        X2.loc[:,'phase3_interval']=X2.loc[:,'phase3_interval']/1000
        scaling_factor=100.0
        if (profile=='small'):
            scaling_factor=1000.0
        #if (profile=='medium'):
        #        scaling_factor=1000.0
        #if (profile=='heavy'):
        #        scaling_factor=100.0
        X2.loc[:,'phase2_torque']=scaling_factor*X2.loc[:,'phase2_torque']
        X2.loc[:,'phase2_torque']=round(X2.loc[:,'phase2_torque'], 1)
        X2.drop_duplicates()

        subprocess.run(['echo', 'Training a model on psm profile: '+profile, ' MAX_TORQUE: ', str(MAX_TORQUE)])
        ###initially we consider all data as negatives...
        y2 = data['init_label']
        y2 = y2.astype(dtype=np.uint8)

        clf = trainModel(X2,y2, scaling_factor, profile)
        
        ###added this just to save the model profile in a file
        filename = 'DSF/model_'+profile+'.sav'
        pickle.dump(clf, open(filename, 'wb'))
        ###loading saved model
        clf = pickle.load(open(filename, 'rb'))

        psm_profiles[profile]=clf
        subprocess.run(['echo', 'Generator trained for psm profile: '+profile])
        #test
        if clf!=None:
            generateRandomTests(clf, 3, X2, scaling_factor)
        #test=generateAdequateTest(clf, X2, scaling_factor)
        #print('Generated test for ', profile)
        #print(test)

####START OF THE MASTER ATG SCRIPT MAIN FUNCTION
def main():
        global psm_profiles
        pd.options.mode.chained_assignment = None

        createResultsFile()

        ######## PHASE 1: TRAIN THE ATG MODEL ###########        
        data = pd.read_csv('code/UC1-RW/all-potential-uc1-inputs-3params.csv', nrows=2744)
        X1 = data[['phase2_torque','phase2_duration','phase3_interval']]
        y1 = data['label']
        
        #####Scale the data in the same climax and discover target boundaries
        X1.loc[:,'phase2_duration']/=1000
        X1.loc[:,'phase3_interval']/=1000
        X1.loc[:,'phase2_torque']=X1.loc[:,'phase2_torque']*10

        ###initially we consider all data as negatives...
        y1 = data['init_label']
        y1 = y1.astype(dtype=np.uint8)

#        clf = trainModel(X1,y1, 1.0, "")
        #print('Generator trained!')
#        subprocess.run(['echo', 'Generator trained!'])

#        generateRandomTests(clf, 10, X1)

        ######## END OF PHASE 1: TRAIN THE ATG MODEL ###########        



        ######## PHASE 2: TRAIN ATG MODEL ON MULTIPLE PSMs PROFILES ###########        

        i=-1
        for profile in psm_profiles.keys():
            i=i+1
            subprocess.run(['echo', 'Simulation ', str(profile)])
            trainOnPSMsize(profile)
            
            
        #wait until all 3 PSM profiles are trained...  
       
        subprocess.run(['echo', 'All three threads have finished...'])

        
        ######## END OF PHASE 2: FURTHER ALIGN THE ATG MODEL ON MULTIPLE PSMs ###########        
        

main()
