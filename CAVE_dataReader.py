# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:13:53 2018

@author: XieQi
"""
import h5py
import os
import numpy as np
import scipy.io as sio  
#from scipy import misc
import MyLib as ML
import random 
#import cv2
def all_train_data_in():
    allDataX = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('CAVEdata/X/'):
           files.sort()
           for j in range(20):
#                print(Ind[0,j])
                i = Ind[0,j]-1
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
#                print(type(inX[1,1,1]))
                allDataX.append(inX)
                
    return allDataX


def all_test_data_in():
    allDataX = []
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('CAVEdata/X/'):
           files.sort()
           for j in range(12):
#                print(Ind[0,j])
                i = Ind[0,j+20]-1
#                print(i)
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX  = data['msi']
                allDataX.append(inX)
    return allDataX

def train_data_in(allX, sizeI, batch_size, channel=31,dataNum = 20):
#    meanfilt = np.ones((32,32))/32/32
    batch_X = np.zeros((batch_size, sizeI, sizeI, channel),'f')
    batch_Z = np.zeros((batch_size, 48, 48, channel),'f')
#    batch_Z = np.zeros((batch_size, sizeI, sizeI, channel))
    for i in range(batch_size):
        ind = random.randint(0, dataNum-1)
        X = allX[ind]
        px = random.randint(0,512-sizeI)
        py = random.randint(0,512-sizeI)
        subX = X[px:px+sizeI:1,py:py+sizeI:1,:]
        
#  随机旋转和反转        
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        for j in range(rotTimes):
            subX = np.rot90(subX)
#            subZ = np.rot90(subZ)
        
        for j in range(vFlip):
            subX = subX[:,::-1,:]
#            subZ = subZ[:,::-1,:]

        
        for j in range(hFlip):
            subX = subX[::-1,:,:]
#            subZ = subZ[::-1,:,:]
        batch_X[i,:,:,:] = subX
#        batch_Z[i,:,:,:] = subZ
    for j in range(2):
        for k in range(2):
            batch_Z = batch_Z + batch_X[:,j:512:2,k:512:2,:]/2/2

    return batch_X, batch_Z


def eval_data_in(batch_size=20):
#    用这两行不需要h5文件
#    allX, allY = all_test_data_in()
#    return train_data_in(allX, allY, 96, 10, 31,12)
#    
    val_h5_X, val_h5_Y, val_h5_Z= read_data('eval1.h5')
    val_h5_X = np.transpose(val_h5_X, (0,2,3,1))   # image X
    val_h5_Y = np.transpose(val_h5_Y, (0,2,3,1))   # image Y
    val_h5_Z = np.transpose(val_h5_Z, (0,2,3,1))/(32*32)   # image 
    rand_index = np.arange(int(0),int(batch_size))
    val_h5_X = val_h5_X[rand_index,:,:,:]
    val_h5_Y = val_h5_Y[rand_index,:,:,:]
    val_h5_Z = val_h5_Z[rand_index,:,:,:]        
    return val_h5_X, val_h5_Y, val_h5_Z

def read_data(file):
    with h5py.File(file, 'r') as hf:
        X = hf.get('X')
        Y = hf.get('Y')
        Z = hf.get('Z')
        return np.array(X), np.array(Y), np.array(Z)


def PrepareDataAndiniValue():
    DataRoad = 'CAVEdata_2/'
    folder = os.path.exists(DataRoad)
    if not folder:
        Ind  = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,32,11,1]; #random index
        data = sio.loadmat('rowData/CAVEdata/response coefficient')
        R    = data['A']
        ML.mkdir(DataRoad+'X/')
        ML.mkdir(DataRoad+'Y/')
        ML.mkdir(DataRoad+'Z/')
        n = 0

        for root, dirs, files in os.walk('rowData/CAVEdata/complete_ms_data/'):
            files.sort()
            for i in range(32):
                n=n+1
                if Ind[i]==32:
                    X = readImofDir('rowData/CAVEdata/complete_ms_data/'+files[Ind[i]-1])/255
                else:
                    X = readImofDir('rowData/CAVEdata/complete_ms_data/'+files[Ind[i]-1])/(2**16-1)
                Y = np.tensordot(X,R,(2,0))
                for j in range(32):
                    for k in range(32):
                        Z = X + X[:,j:512:32,k:512:32,:]/32/32
                sio.savemat(DataRoad+'X/'+files[Ind[i]-1], {'msi': X})     
                sio.savemat(DataRoad+'Y/'+files[Ind[i]-1], {'RGB': Y})   
                sio.savemat(DataRoad+'Z/'+files[Ind[i]-1], {'Zmsi': Z})   
                if n<=20:
                    if n==1:
                        allX = np.reshape(X,[512*512,31])
                        allY = np.reshape(Y,[512*512,3])
                    else:
                        allX = np.hstack((allX,np.reshape(X,[512*512,31])))
                        allY = np.hstack((allY,np.reshape(Y,[512*512,3])))       
        allX = np.matrix(allX)
        allY = np.matrix(allY)
        iniA = (allY.T*allY).I*(allY.T*allX)
        sio.savemat(DataRoad, {'iniA': iniA}) 
        
        initemp = np.eye(31)
        iniUp1 = np.tile(initemp,[3,3,1,1])
        sio.savemat(DataRoad, {'iniUp': iniUp1}) 
    else:
        print('Using the prepared data and initial values in folder CAVEdata')
        
def readImofDir(theRoad):
    
    I = cv2.imread('./cc_1.png')
    return X
        
#  
def generateZ():
    import os
    import numpy as np
    import scipy.io as sio
    for root, dirs, files in os.walk('CAVEdata/X/'):
        files.sort()
        for j in range(32):
            data = sio.loadmat("CAVEdata/X/"+files[j])
            inX  = data['msi']
            Z = np.zeros((128, 128, 31),'f')
            for l in range(4):
                for k in range(4):
                    Z = Z + inX[l:512:4,k:512:4,:]/4/4
            sio.savemat("CAVEdata/Z_factor4/"+files[j], {'Zmsi': Z})  
    

                        
                
                    
                
            
        
        

            
    


#
#allX, allY = all_train_data_in()
#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
#batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#print(type(batch_X[1,1,1,1]))
#for nframe in range(20) :
##    nframe = 18
#    print(nframe)
#    X = batch_X[nframe,:,:,:]
#    Y = batch_Y[nframe,:,:,:]
#    Z = batch_Z[nframe,:,:,:]
#    toshow = np.hstack((ML.normalized(X[:,:,[0,15,30]]),ML.normalized(Y[:,:,[0,1,2]])))
#    ML.imshow(toshow)
#    ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))
#
#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
##batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#X = batch_X[0,:,:,:]
#Y = batch_Y[0,:,:,:]
#Z = batch_Z[0,:,:,:]
#toshow = np.hstack((ML.normalized(X[:,:,[0,1,2]]),Y[:,:,[0,1,2]]))
#ML.imshow(toshow)
#ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))
#
#batch_X, batch_Y, batch_Z= train_data_in(allX, allY, 96, 10, 31)
##batch_X, batch_Y, batch_Z= eval_data_in()    
#print(batch_X.shape)
#X = batch_X[0,:,:,:]
#Y = batch_Y[0,:,:,:]
#Z = batch_Z[0,:,:,:]
#toshow = np.hstack((ML.normalized(X[:,:,[0,1,2]]),Y[:,:,[0,1,2]]))
#ML.imshow(toshow)
#ML.imshow(ML.normalized(Z[:,:,[0,1,2]]))