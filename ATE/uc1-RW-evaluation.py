import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
import os
from pathlib import Path  

def plot(x,y, labelX, labelY):
    plt.plot(x, y)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.show()
    return

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


data = []
test_case=4
with open('test_output'+str(test_case)+'.csv') as file:
    for line in file:
        line = line.strip()
        data.append(line.split())

print('=====================================')
print('Evaluating WHL50 TEST-', test_case)

df = pd.DataFrame(data)

x=df.index.values.tolist() 
y=df.iloc[:, 1]
y=y.astype(dtype=float)
yVel=y

print('=====================================')
max_vel=0
peak_index=0
stop_index=0
###Identify maximum speed
for index, vel in y.items():
    if vel<max_vel:  #check if velocity just decreased at this timepoint
        print('maximum velocity= ', max_vel, 'at time: ', peak_index)
        break
    else:
        max_vel=vel
        peak_index=index

###Identify stop timepoint AFTER maximum speed
for index, vel in y.items():
    if (index>peak_index):
        stop_index=index
        if vel==0:
            print('stopped at timepoint= ', stop_index)
            break

passive_start_index=peak_index
### Identify timepoint for passive run-down
y_torApplication=df.iloc[:, 0]
y_torApplication=y_torApplication.astype(dtype=float)
torque_applied=0
for index, tor in y_torApplication.items():
    if index>peak_index:
        if (tor==1):
            passive_start_index=index
        if (tor==0):
            break

tts=stop_index-passive_start_index
print('Time to stop = ', tts, 'ms')

print('=====================================')


### Calculate friction using equation: F= sign(v)*(Gain * abs(v) + Offset)


spec_file_path ="../Run pipeline outputs/DSF/database/"
spec_file_name = "DSF_SDB_WHL50.xls"
gain = getValueFromSpecs(spec_file_path+spec_file_name, 'GAIN')
offset = getValueFromSpecs(spec_file_path+spec_file_name, 'OFFSET')
    
time = x
friction = y
for index, vel in y.items():
    friction[index] = np.sign(vel)*(gain*abs(vel)+offset)
#plot(time, friction, 'Time', 'Friction')
gain=f'{gain:.9f}'
offset=f'{offset:.6f}'
print('SPEC values offset: ', offset, ', gain: ', gain)


#####Try to estimate values ONLY with the speed curve (for real RW)
yVel=df.iloc[:, 1]
yVel=yVel.astype(dtype=float)

inertia = getValueFromSpecs(spec_file_path+spec_file_name, 'INERTIA')
MAX_TORQUE = getValueFromSpecs(spec_file_path+spec_file_name, 'MAX_TORQUE')

print('peak_index=', peak_index, 'passive_start_index=', passive_start_index, ' stop_index=', stop_index, ' inertia=', inertia, ' max_vel=',max_vel, ' MAX_TORQUE=',MAX_TORQUE)
plot(x,yVel, 'Time', 'outRWS_Speed')


prev_index=0
prev_vel=max_vel
yDec=yVel
yDecSmooth=yDec
######First calculate decceleration curve yDec=OutTorque=I*Dv/dt 
for index, vel in yVel.items():
    if index<passive_start_index or index>=stop_index:
        yDec[index]=0.0
        yDecSmooth[index]=0.0
        prev_vel=vel
        prev_index=index
        continue

    dt=index-prev_index
    yDec[index]=inertia*(vel-prev_vel)/dt
    yDecSmooth[index]=yDec[index]
####Check deltas
    if abs(yDec[index])>MAX_TORQUE:
        yDecSmooth[index]=0.0
        print('Friction Delta observed at time: ', str(index), ' Possible fault in RW!!')
    prev_index=index
    prev_vel=vel
plot(x,yDec, 'Time', 'Decceleration_torque')

######Smooth Decceleration_torque
for index, yb in yDec.items():
    if (index>passive_start_index) and (index<stop_index) and (abs(yb)<0.000000000001):
        print('Friction value=0 observed at time: ', str(index), ' Possible fault in RW!!')
        yDecSmooth[index]=yDec[prev_index]
    if (index>stop_index):
        yDecSmooth[index]=0.0
    prev_index=index
plot(x,yDecSmooth, 'Time', 'Decceleration_torque_smoothed')

######Calculate Gain=dDec/dV, offset=yDec-V*Gain
prev_tor=0.0
prev_index=0
A=0.0
B=0.0
sumA = 0.0
sumB = 0.0
#xa = x
yVel=df.iloc[:, 1]
yVel=yVel.astype(dtype=float)
ya = yDec
samples=0
for index, tor in yDec.items():
    ######Ignoring serious turbulance in the beginning and the end of passive rundown
    if index<=passive_start_index+23 or index>=stop_index-13:
        ya[index]=0.0
        prev_tor=tor
        prev_index=index
        continue
    dTor=tor-prev_tor
    dt=index-prev_index
    if dt==0:
        dt=1

    dVel=yVel[index]-yVel[prev_index]
    if dVel==0:
        A=0
    else:
        A=dTor/dVel
   
    A=round(A,13)
    sumA=sumA+A
    B=abs(tor)-abs(A)*yVel[index]
    B=round(B,13)
    sumB=sumB+B
    ya[index]=A

    samples=samples+1
    prev_tor=tor
    prev_index=index

plot(x,ya, 'time', '(dDec/dV)')
est_gain=abs(sumA/float(samples))
est_gain=f'{est_gain:.9f}'
est_offset=abs(sumB/float(samples))
est_offset=f'{est_offset:.6f}'
print('Using output speed to estimate offset:', est_offset, 'gain:', est_gain)

print('Evaluation finished...')    
print('=====================================')
