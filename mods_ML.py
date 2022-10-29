# -*- coding: utf-8 -*-

# Description:
# 程序功能：训练模型及对模型做可解释性分析
#       Step1：根据学习时间删除相应的参数
#       Step2：数据标准化处理，根据患者ID划分训练集测试集，并运用Lightbgm方法训练
#       Step3：找出敏感性和特异性相差最小的点即为最优分类阈值，在本项目中并没用到此项计算方法，所以设置默认值为50
#       Step4：根据预测概率与最优分类阈值（cut off值）对患者进行生死预测，并计算预测结果的各项指标（如需要置信区间，则继续运行程序）
# 程序运行结果：输出预测结果的各项指标
#
# DataFile：数据为动态处理方法后得到的结果
#  
# Output：
#        eva_comm:预测结果的各项指标
#
# v1.0 2022/5/12



from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pylab as plt
import lightgbm as lgb

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn    import  svm
import lightgbm as lgb
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn import svm
from sklearn import neighbors
from numpy import *
import seaborn as sns

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)


def evaluating_indicator(y_true, y_test, y_test_value):  #计算预测结果的各项指标
    c_m = confusion_matrix(y_true, y_test)
    TP=c_m[0,0]
    FN=c_m[0,1]
    FP=c_m[1,0]
    TN=c_m[1,1]
    
    TPR=TP/ (TP+ FN) #敏感性
    TNR= TN / (FP + TN) #特异性
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_true, y_test)
    MCC = matthews_corrcoef(y_true, y_test)
    F1score =  f1_score(y_true, y_test)
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    KAPPA=kappa(c_m)
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC,'KAPPA':KAPPA}
    return c

def blo(pro_comm_Pre,jj):     #根据预测概率与最优分类阈值（cut off值）对患者进行生死预测
    blo_Pre=zeros(len(pro_comm_Pre)) 
    blo_Pre[(pro_comm_Pre[:,1]>(jj*0.01))]=1
    return blo_Pre

def spec_for_ser(df,icustay_id):  ##根据患者ID划分训练集测试集（环境不一样,所以自己写）
    str_df=str(df)
    for i in icustay_id:
        if i==icustay_id[0]:  ##如果i在icustay_id这个形参中
            input_mulit=(str_df+"["+str_df+"['icustay_id']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['icustay_id']=={}]".format(i))
    return (pd.concat(eval(input_mulit),axis=0,ignore_index=True))


comtest = pd.read_csv("/home/amms229/Liu/gj_0329p_step2_nodeath.csv")#,nrows=200000

comtest.drop(['hadm_id'],axis=1,inplace=True)  ## 全部参数
comtest.drop(['sofa_score_1','marshll_score_1','qsofa_score_1','lods_score_1'],axis=1,inplace=True)#

comtest.drop(['urea_1','epinephrine_1','norepinephrine_1','dobutamine_1','dopamine_1','bilirubin_1','creatinine_1','platelet_1'],axis=1,inplace=True) ##实验室参数和无创参数
comtest.drop(['ph_1','pf_ratio_1','par_1','bicarbonate_bg_1','totalco2_bg_1','chloride_bg_1','calcium_bg_1','hematocrit_bg_1','glucose_bg_1','pco2_bg_1','po2_bg_1','potassium_bg_1','bun_lab_1', 'wbc_lab_1','aniongap_lab_1','bicarbonate_lab_1','chloride_lab_1','glucose_lab_1','hematocrit_lab_1','potassium_lab_1','ptt_lab_1','inr_lab_1','pt_lab_1','sodium_lab_1','mchc_else_1','mch_else_1','mcv_else_1','red_blood_cells_else_1','magnesium_else_1','phosphate_else_1','calcium_total_else_1','hemoglobin_1'],axis=1,inplace=True)
#comtest.drop(['gender','fio2_1','spo2_1','vent_1','uo_1','uosum_1','gcs_1','meanbp_1','gcsmotor_1','gcsverbal_1','gcseyes_1','heartrate_1','sysbp_1','diasbp_1','resprate_1','glucose_1','temperature_1'],axis=1,inplace=True)
#comtest.drop(['gender','fio2_1','spo2_1','vent_1','uo_1','uosum_1','meanbp_1','gcsmotor_1','gcsverbal_1','gcseyes_1','heartrate_1','diasbp_1','glucose_1','temperature_1'],axis=1,inplace=True)
icustay_id=list(set(comtest['icustay_id']))

scaler = StandardScaler()   #对病例数据进行标准化处理，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
comtest.iloc[:,1:comtest.shape[1]-1]=scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]-1]) ##先拟合数据，再标准化
   
#根据患者ID划分训练集测试集
x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.2,random_state=1)    

x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);
y_train_for_vail=x_train_for_vail.iloc[:,-1];
x_train_for_vail_group=x_train_for_vail.iloc[:,0];
x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
x_test=spec_for_ser('comtest',x_test);
y_true=x_test.iloc[:,-1];
x_test_group=x_test.iloc[:,0];
x_test=x_test.iloc[:,1:x_test.shape[1]-1]


comm = lgb.LGBMClassifier()


comm.fit(x_train_for_vail ,y_train_for_vail)
print('wuchuang  mod nb')
pro_comm_Pre = comm.predict_proba(x_test)

print('pro_comm_Pre done')
# blo_comm_Pre = blo(pro_comm_Pre,50)  ##敏感性和特异性相差最小的点

# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
# print(eva_comm) 

##计算UMC中敏感性和特异性相差的最小值(cut_off)
RightIndex=[]
for jj in range(100): #计算模型在不同分类阈值下的各项指标
    blo_comm_Pre = blo(pro_comm_Pre,jj)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出
position=position.mean()

blo_comm_Pre = blo(pro_comm_Pre,position)  ##敏感性和特异性相差最小的点

eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
print(eva_comm)  ##常规计算



#计算模型性能95%置信区间  
y_true=np.array(y_true)
n_bootstraps = 2000

bootstrapped_scores=[]
EVA_ACC_CI=[];EVA_AUC_CI=[]
EVA_MCC_CI=[];EVA_F1_score_CI=[]
EVA_BER_CI=[];EVA_KAPPA_CI=[]
EVA_TPR_CI=[];EVA_TNR_CI=[]

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices

    indices=np.random.randint(0,len(pro_comm_Pre[:,1]) - 1,len(pro_comm_Pre[:,1]))
    if len(np.unique(y_true[indices])) < 2:
        # We need at least one positive and one negative sample for ROC AUC
        # to be defined: reject the sample
        continue

    ### score = roc_auc_score(y_true[indices], y_pred[indices])
    eva_CI = evaluating_indicator(y_true=y_true[indices], y_test=blo_comm_Pre[indices], y_test_value=pro_comm_Pre[indices])
    EVA_AUC_CI.append(eva_CI['AUC']);
    EVA_ACC_CI.append(eva_CI['ACC']);
    EVA_MCC_CI.append(eva_CI['MCC']);
    EVA_F1_score_CI.append(eva_CI['F1_score']);EVA_BER_CI.append(eva_CI['BER']);EVA_KAPPA_CI.append(eva_CI['KAPPA']);
    EVA_TPR_CI.append(eva_CI['TPR']);EVA_TNR_CI.append(eva_CI['TNR']);
    ### print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
   
sorted_AUC_scores = np.array(EVA_AUC_CI); sorted_AUC_scores.sort()          
sorted_ACC_scores = np.array(EVA_ACC_CI); sorted_ACC_scores.sort()
sorted_MCC_scores = np.array(EVA_MCC_CI); sorted_MCC_scores.sort()
sorted_F1_score_scores = np.array(EVA_F1_score_CI); sorted_F1_score_scores.sort()
sorted_BER_scores = np.array(EVA_BER_CI); sorted_BER_scores.sort()
sorted_KAPPA_scores = np.array(EVA_KAPPA_CI); sorted_KAPPA_scores.sort()
sorted_TPR_scores = np.array(EVA_TPR_CI); sorted_TPR_scores.sort()
sorted_TNR_scores = np.array(EVA_TNR_CI); sorted_TNR_scores.sort()





# Computing the lower and upper bound of the 90% confidence interval
# You can change the bounds percentiles to 0.025 and 0.975 to get
# a 95% confidence interval instead.
print("Confidence interval for the AUC: [{:0.6f} - {:0.6}]".format(sorted_AUC_scores[int(0.025 * len(sorted_AUC_scores))], sorted_AUC_scores[int(0.975 * len(sorted_AUC_scores))]))
print("Confidence interval for the ACC: [{:0.6f} - {:0.6}]".format(sorted_ACC_scores[int(0.025 * len(sorted_ACC_scores))], sorted_ACC_scores[int(0.975 * len(sorted_ACC_scores))]))
print("Confidence interval for the BER: [{:0.6f} - {:0.6}]".format(sorted_BER_scores[int(0.025 * len(sorted_BER_scores))], sorted_BER_scores[int(0.975 * len(sorted_BER_scores))]))
print("Confidence interval for the F1_score: [{:0.6f} - {:0.6}]".format(sorted_F1_score_scores[int(0.025 * len(sorted_F1_score_scores))], sorted_F1_score_scores[int(0.975 * len(sorted_F1_score_scores))]))
print("Confidence interval for the KAPPA: [{:0.6f} - {:0.6}]".format(sorted_KAPPA_scores[int(0.025 * len(sorted_KAPPA_scores))], sorted_KAPPA_scores[int(0.975 * len(sorted_KAPPA_scores))]))
print("Confidence interval for the MCC: [{:0.6f} - {:0.6}]".format(sorted_MCC_scores[int(0.025 * len(sorted_MCC_scores))], sorted_MCC_scores[int(0.975 * len(sorted_MCC_scores))]))
print("Confidence interval for the TNR: [{:0.6f} - {:0.6}]".format(sorted_TNR_scores[int(0.025 * len(sorted_TNR_scores))], sorted_TNR_scores[int(0.975 * len(sorted_TNR_scores))]))
print("Confidence interval for the TPR: [{:0.6f} - {:0.6}]".format(sorted_TPR_scores[int(0.025 * len(sorted_TPR_scores))], sorted_TPR_scores[int(0.975 * len(sorted_TPR_scores))]))
##################################################################################
