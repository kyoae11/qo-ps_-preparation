import numpy as np
import pandas as pd
import random
from afil.paths import DATA_DIR

def matrix2sequence(data):

    np.set_printoptions(suppress=True)
    n,m=data.shape
    data_list=np.zeros((m*n,3))
    x=np.tile(np.arange(0,n).reshape(-1,1),(1,m)).flatten()
    y=np.tile(np.arange(0,m).reshape(1,-1),(n,1)).flatten()
    data_list[:,0]=x
    data_list[:,1]=y
    data_list[:,2]=data.flatten()
    return data_list

def removeEntries(matrix, density, seedID):
	(vecX, vecY) = np.where(matrix > 0)
	vecXY = np.c_[vecX, vecY]
	numRecords = vecX.size
	numAll = matrix.size
	random.seed(seedID)
	randomSequence = np.arange(0, numRecords)
	random.shuffle(randomSequence) # one random sequence per round
	numTrain = int(numAll * density)
	# by default, we set the remaining QoS records as testing data
	numTest = numRecords - numTrain
	trainXY = vecXY[randomSequence[0 : numTrain], :]
	testXY = vecXY[randomSequence[- numTest :], :]

	trainMatrix = np.zeros(matrix.shape)
	trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
	testMatrix = np.zeros(matrix.shape)
	testMatrix[testXY[:, 0], testXY[:, 1]] = matrix[testXY[:, 0], testXY[:, 1]]

    # ignore invalid testing data
	idxX = (np.sum(trainMatrix, axis=1) == 0)
	testMatrix[idxX, :] = 0
	idxY = (np.sum(trainMatrix, axis=0) == 0)
	testMatrix[:, idxY] = 0
	return trainMatrix, testMatrix

def get_list(mat,density,seedID):
    train_mat, test_mat = removeEntries(mat,density,seedID)
    train_list,test_list=matrix2sequence(train_mat),matrix2sequence(test_mat)
    train_id,test_id=np.where(train_list[:,2]!=0),np.where(test_list[:,2]!=0)
    train_list,test_list=train_list[train_id],test_list[test_id]
    return train_list,test_list

if __name__ == '__main__':
    mat_path = DATA_DIR / 'rtMatrix.txt'
    mat = np.array(pd.read_table(mat_path, header=None).iloc[:, :-1])
    a,b=get_list(mat,0.2,0)
    print(a,b)
