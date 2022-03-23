import os
import numpy as np
import random
import plotly.graph_objects as go
import scipy
from libsvm.svmutil import *
import matplotlib.pyplot as plt
nums = ["one", "two", "three", "four", "five"]
# change label of data to desired classification labels and output to traindata and testdata
trainlines = []
testlines = []
with open("Abalone.txt", 'r+') as datafile:
    lines = datafile.readlines()
    for i in range(4177):
        l = lines[i].split(' ')
        label = "-1" if (int(l[0])<=9) else "1"
        l[0] = label
        if i < 3133:
            trainlines.append(" ".join(l))
        else:
            testlines.append(" ".join(l))
with open("train.txt",'w') as trainfile:
    trainfile.writelines(trainlines)
with open("test.txt",'w') as testfile:
    testfile.writelines(testlines)
# scale data correspondingly
os.system("./svm-scale -l -1 -u 1 -s range train.txt > scaledtrain.txt")
os.system("./svm-scale -r range test.txt > scaledtest.txt")
# shuffle lines traning data and split into disjoint sets
with open("scaledtrain.txt",'r+') as ts:
    lines = ts.readlines()
    random.shuffle(lines)
    with open("trainone.txt",'w') as trainone:
        trainone.writelines(lines[:627])
    with open("traintwo.txt",'w') as traintwo:
        traintwo.writelines(lines[627:1254])
    with open("trainthree.txt",'w') as trainthree:
        trainthree.writelines(lines[1254:1881])
    with open("trainfour.txt",'w') as trainfour:
        trainfour.writelines(lines[1881:2507])
    with open("trainfive.txt",'w') as trainfive:
        trainfive.writelines(lines[2507:3133])
    # create corresponding testing sets
    with open("testone.txt",'w') as testone:
        testone.writelines(lines[627:])
    with open("testtwo.txt",'w') as testtwo:
        testtwo.writelines(lines[:627]+lines[1254:])
    with open("testthree.txt",'w') as testthree:
        testthree.writelines(lines[:1254]+lines[1881:])
    with open("testfour.txt",'w') as testfour:
        testfour.writelines(lines[:1881]+lines[2507:])
    with open("testfive.txt",'w') as testfive:
        testfive.writelines(lines[:2507])
# train and cross-validate data
k = 10
avgs = []
plussd = []
minussd = []
sd = []
rg = np.arange(-k,k+1)
for exp in rg:
    c = str(3.0**exp)
    errors = []
    for i in range(5):
        train_y, train_x = svm_read_problem("train%s.txt"%nums[i])
        test_y, test_x = svm_read_problem("test%s.txt"%nums[i])
        m = svm_train(train_y, train_x, "-t 1 -d %s -c %s -q"%(str(i+1),c))
        label,acc,val = svm_predict(test_y, test_x, m, "-q")
        errors.append(acc[1])
    avgs.append(np.average(errors))
    sd.append(np.std(errors))
for i in range(len(avgs)):
    plussd.append(avgs[i]+sd[i])
    minussd.append(avgs[i]-sd[i])
print(avgs)
# plot data
plt.plot(rg,plussd,label="+1 standard deviation")
plt.plot(rg,avgs, label = "average cross-validation error")
plt.plot(rg,minussd, label = "-1 standard deviation")
plt.title("Average Cross-Validation Error ±1 Standard Deviation vs. Cost")
plt.xlabel("lg(Cost)")
plt.ylabel("Cross-Validation Error")
plt.legend()
plt.show()
# fix c_pime
c_prime = str(3**7)
d = np.arange(1,6)
fh_errors = []
test_errors = []
test_y, test_x = svm_read_problem("scaledtest.txt")
for i in range(5):
    train_y, train_x = svm_read_problem("train%s.txt"%nums[i])
    crosstest_y, crosstest_x = svm_read_problem("test%s.txt"%nums[i])
    m = svm_train(train_y, train_x, "-t 1 -d %s -c %s -q"%(str(i+1),c_prime))
    fhlabel,fhacc,fhval = svm_predict(crosstest_y, crosstest_x, m, "-q")
    testlabel,testacc,testval = svm_predict(test_y, test_x, m, "-q")
    fh_errors.append(fhacc[1])
    test_errors.append(testacc[1])
plt.plot(d,fh_errors,label = "five-hold cross-validation error")
plt.plot(d,test_errors,label = "test errors")
plt.title("Error vs. d")
plt.xlabel("d")
plt.ylabel("Error")
plt.legend()
plt.show()
# fix d_prime
d_prime = "3"
train_y, train_x = svm_read_problem("scaledtrain.txt"%nums[i])
test_y, test_x = svm_read_problem("test%s.txt"%nums[i])
m = svm_train(train_y, train_x, "-t 1 -d %s -c %s -q"%(d_prime,c))
label,acc,val = svm_predict(test_y, test_x, m, "-q")
plt.plot(d,fh_errors,label = "five-hold cross-validation error")
plt.plot([1],acc[1],label = "test errors")
plt.ylabel("Test Error")
plt.legend()
plt.show()
