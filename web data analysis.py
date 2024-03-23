# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:05:28 2018

@author: gyang
"""
import pyodbc as po
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import graphviz
import matplotlib.pyplot as plt
!pip install graphviz 
import os
from sklearn.datasets import load_iris
iris = load_iris()
iris.feature_names
iris.target_names
iris.target.shape
os.environ["PATH"] += os.pathsep + 'D:\\Guohui\\Program Files (x86)\\Graphviz2.38\\bin\\'
np.set_printoptions(threshold=50)
conn=po.connect(dsn="Netezza 6",uid='gyang',pwd='Y<87814461')
web_email_query1421='''select b.*, case when c.emails_clicked>0 then c.emails_clicked else 0 end as num_emails,case when web.avg>0 then web.avg else 0 end as SCORE from
                    NZPDB003..GY_ID1421 as b
                    left join
                    (select vc.edw_hshld_num, sum(Emails_clicked) as Emails_clicked from
                                     lehub.vw_consumer VC
                    	             
                             inner join 
                    	           (select c.CUR_CNSMR_NUM, count(distinct eo.EMAIL_OFFR_NUM || eo.CAMP_EVNT_NUM || oce.EVNT_DT) as Emails_clicked
                    		   FROM LEHUB.CONSUMER                  C
                    		    inner join LEHUB.CUST_CNTCT_PT           CCP
                    				on C.CUST_ID = CCP.CUST_ID
                    			inner join LEHUB.OUTBND_CUST_EVNT_CUST_CNTCT_PT           OCECP
                    				on CCP.CUST_CNTCT_PT_NUM = OCECP.CUST_CNTCT_PT_NUM
                    		    inner join  LEHUB.OUTBND_CUST_EVNT        OCE
                    				on OCECP.OUTBND_CUST_EVNT_NUM = OCE.OUTBND_CUST_EVNT_NUM
                    				and OCE.EVNT_TYP_CD  = OCECP.EVNT_TYP_CD
                    			inner join LEHUB.EMAIL_OFFR              EO
                    				on OCE.OUTBND_CUST_EVNT_NUM = EO.OUTBND_CUST_EVNT_NUM
                    				and eo.EVNT_TYP_CD = oce.EVNT_TYP_CD
                    			inner join lehub.EMAIL_OFFR_RSLT_DTL_AGG DA
                    				on DA.OUTBND_CUST_EVNT_NUM = OCE.OUTBND_CUST_EVNT_NUM
                    				and da.EVNT_TYP_CD = OCE.EVNT_TYP_CD
                    		  where  oce.EVNT_TYP_CD = 1 and oce.EVNT_METH_CD = 6 and oce.BUS_UNIT_NUM = 1 and DA.RSLT_CD in (10) and DA.RSLT_DT between '2018-02-06' AND '2018-02-27'
                    		  group by 1) A
                            on A.cur_cnsmr_num = vc.cur_cnsmr_num
                             group by 1
                    
                    
                    ) c
                    
                    on b.hh_num=c.edw_hshld_num
                    left join                    
                    nzpdb003.rmddev_user.gyang_web_browser as web
                    on b.hh_num=web.EDW_hshld_num'''
web_email1421=pd.read_sql_query(web_email_query1421,conn)
web_email_query1422='''select b.*, case when c.emails_clicked>0 then c.emails_clicked else 0 end as num_emails,case when web.avg>0 then web.avg else 0 end as SCORE from
                    NZPDB003..GY_ID1422 as b
                    left join
                    (select vc.edw_hshld_num, sum(Emails_clicked) as Emails_clicked from
                                     lehub.vw_consumer VC
                    	             
                             inner join 
                    	           (select c.CUR_CNSMR_NUM, count(distinct eo.EMAIL_OFFR_NUM || eo.CAMP_EVNT_NUM || oce.EVNT_DT) as Emails_clicked
                    		   FROM LEHUB.CONSUMER                  C
                    		    inner join LEHUB.CUST_CNTCT_PT           CCP
                    				on C.CUST_ID = CCP.CUST_ID
                    			inner join LEHUB.OUTBND_CUST_EVNT_CUST_CNTCT_PT           OCECP
                    				on CCP.CUST_CNTCT_PT_NUM = OCECP.CUST_CNTCT_PT_NUM
                    		    inner join  LEHUB.OUTBND_CUST_EVNT        OCE
                    				on OCECP.OUTBND_CUST_EVNT_NUM = OCE.OUTBND_CUST_EVNT_NUM
                    				and OCE.EVNT_TYP_CD  = OCECP.EVNT_TYP_CD
                    			inner join LEHUB.EMAIL_OFFR              EO
                    				on OCE.OUTBND_CUST_EVNT_NUM = EO.OUTBND_CUST_EVNT_NUM
                    				and eo.EVNT_TYP_CD = oce.EVNT_TYP_CD
                    			inner join lehub.EMAIL_OFFR_RSLT_DTL_AGG DA
                    				on DA.OUTBND_CUST_EVNT_NUM = OCE.OUTBND_CUST_EVNT_NUM
                    				and da.EVNT_TYP_CD = OCE.EVNT_TYP_CD
                    		  where  oce.EVNT_TYP_CD = 1 and oce.EVNT_METH_CD = 6 and oce.BUS_UNIT_NUM = 1 and DA.RSLT_CD in (10) and DA.RSLT_DT between '2018-02-06' AND '2018-02-27'
                    		  group by 1) A
                            on A.cur_cnsmr_num = vc.cur_cnsmr_num
                             group by 1
                    
                    
                    ) c
                    
                    on b.hh_num=c.edw_hshld_num
                    left join                    
                    nzpdb003.rmddev_user.gyang_web_browser as web
                    on b.hh_num=web.EDW_hshld_num'''
consum='select LEHUB.CONSUMER 
web_email1422=pd.read_sql_query(web_email_query1422,conn)
web_email1422.ID_NUM[:6]
np.where(web_email1422.columns.values=='ADJ_DIR_PROFIT')
column=web_email1422.columns.values
new_column=np.c_[column[[0,1,2,11,-1,-2]].reshape(1,-1),column[3:11].reshape(1,-1),column[12:-2].reshape(1,-1)]
new_column2=list(np.squeeze(new_column))
column[3:11].reshape(1,-1)
column[12:-2]
column[[0,1,2,11,-1,-2],3:11].shape
column[0,1]
column=column[]
web_email1421=web_email1421.loc[:,new_column2]
web_email1421.head(3)
web_email1421=web_email1421.apply(lambda x:x.replace(np.nan,0))
web_email1422=web_email1422.loc[:,new_column2]
web_email1422=web_email1422.replace(np.nan,0)
web_email1421.apply(lambda x:x.isnull()).any()
web_email1421.shape
web_email1422.SCORE.unique()
web_email1421.KGROUP.unique()
web_email1422[web_email1422.KPANEL==1].SCORE.unique()
web_email1421_modify=web_email1421.iloc[:,[1,2,4,5,6,-1,-7]]
web_email1422_modify=web_email1422.iloc[:,[1,2,3,12,18,-1,-2]]
web_email1422_modify.columns
web_email1421_modify.SCORE.nunique()
web_email1421_modify['SCORE0_1']=np.where(web_email1421_modify.SCORE>0,1,0)
web_email1422_modify['SCORE0_1']=np.where(web_email1422_modify.SCORE>0,1,0)
web_email1421_modify['NUM_EMAILS0_1']=np.where(web_email1421_modify.NUM_EMAILS>0,1,0)
web_email1422_modify['NUM_EMAILS0_1']=np.where(web_email1422_modify.NUM_EMAILS>0,1,0)
pd.crosstab(web_email1421.KGROUP,web_email1421.KPANEL,values=web_email1421.SCORE,aggfunc='count')
web_email1421_modify.groupby(['KPANEL','NUM_EMAILS0_1']).NUM_EMAILS0_1.count()
web_email1422_modify.groupby(['KPANEL','NUM_EMAILS0_1']).NUM_EMAILS0_1.count()
web_email1421_modify.groupby(['KGROUP','SCORE0_1']).SCORE0_1.count()
web_email1422_modify.groupby(['KGROUP','SCORE0_1']).SCORE0_1.count()
test1422_emailable=web_email1422[web_email1422.KGROUP==1][['SCORE','NUM_EMAILS','DIRECT_RESP']]
X1=test1422_emailable[['SCORE','NUM_EMAILS']]
list(test1421_emailable.columns.values[:2])
y1=test1422_emailable[['DIRECT_RESP']]
y1.iloc[0,0]
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,train_size=0.7,random_state=1)
np.set_printoptions(threshold=50)
depth=10
result=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree1=DecisionTreeClassifier(max_depth=i,min_samples_split=5,min_samples_leaf=5)
    tree1_model=tree1.fit(X1_train,y1_train)
    pred_train=tree1.predict_proba(X1_train)
    pred_test=tree1.predict_proba(X1_test)
    result[i-2,1]=roc_auc_score(y1_train,pred_train[:,1])
    result[i-2,2]=roc_auc_score(y1_test,pred_test[:,1])
    result[i-2,0]=i
result
tree_best=DecisionTreeClassifier(max_depth=5,min_samples_split=5,min_samples_leaf=5)
tree1_best_model=tree_best.fit(X1_train,y1_train)
train_score=tree_best.predict_proba(X1_train)
tree_best.feature_importances_
gb_imp=pd.DataFrame({'variable':X1_train.columns,'importance':tree_best.feature_importances_}).sort_values(by='importance',ascending=False)
plt.barh(range(len(gb_imp['variable'])),gb_imp['importance'])
plt.yticks(range(len(gb_imp['variable'])), gb_imp['variable'])
for a,b in zip(gb_imp['importance'],range(len(gb_imp['variable']))): 
            plt.text(a+0.001, b, str(round(a,3)),fontsize=12)
plt.xlabel('Relative Importance')
plt.title('Top Important Variable of Tree')
plt.show()
dot_data = export_graphviz(tree1_best_model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("test1421_emailable") 
dot_data = export_graphviz(tree1_best_model, out_file=None, 
                         feature_names=list(test1421_emailable.columns.values[:2]),  
                         class_names=np.array(['no_resp','resp']),  
                         filled=True, rounded=True, proportion=True, 
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph 
p00=315254/319511
p01=4297/319511
p100=289605/293190
p101=3585/293190
p110=25649/26361
p111=712/26361
p200=286246/289752
p201=3506/289752
p210=3359/3438
p211=79/3438
p220=16349/16820
p221=381/16280
p230=9210/9541
p200,p201,p210,p211,p220,p221,p230
Yp231=331/9541
286246/319551
3506/319551
3359/319551
79/319511
test1421_emailable.columns.values[2]
159/7825
4/314
19/935
3454/289972
web_email1422.columns
test1422_emailable_dmd=web_email1422[web_email1422.KGROUP==1][['SCORE','NUM_EMAILS','DIRECT_DMD']]
X2=test1422_emailable_dmd[['SCORE','NUM_EMAILS']]

y2_dmd=test1422_emailable_dmd[['DIRECT_DMD']]
 
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2_dmd,train_size=0.7,random_state=1)
np.set_printoptions(threshold=50)
depth=10
result2=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree2=DecisionTreeRegressor(max_depth=i,min_samples_split=4,min_samples_leaf=4)
    tree2_model=tree2.fit(X2_train,y2_train)
    pred_train=tree2.predict(X2_train)
    pred_test=tree2.predict(X2_test)
    R_train=tree2.score(X2_train,y2_train)
    result2[i-2,1]=tree2.score(X2_train,y2_train)
    result2[i-2,2]=tree2.score(X2_test,y2_test)
    result2[i-2,0]=i
result2
tree_best2=DecisionTreeRegressor(max_depth=3,min_samples_split=5,min_samples_leaf=5)
tree2_best_model=tree_best2.fit(X2_train,y2_train)
train_score2=tree_best.predict_proba(X2_train)

tree_best2.feature_importances_
gb_imp=pd.DataFrame({'variable':X2_train.columns,'importance':tree_best2.feature_importances_}).sort_values(by='importance',ascending=False)
plt.barh(range(len(gb_imp['variable'])),gb_imp['importance'])
plt.yticks(range(len(gb_imp['variable'])), gb_imp['variable'])
for a,b in zip(gb_imp['importance'],range(len(gb_imp['variable']))): 
            plt.text(a+0.001, b, str(round(a,3)),fontsize=12)
plt.xlabel('Relative Importance')
plt.title('Top Important Variable of Tree')
plt.show()
dot_data = export_graphviz(tree2_best_model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("test1422_emailable_dmd") 
dot_data = export_graphviz(tree2_best_model, out_file=None, 
                         feature_names=list(test1422_emailable.columns.values[:2]),  
                         class_names=np.array(['demand']),  
                         filled=True, rounded=True,
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph 
web_email1422.shape
####Tree for 1422
test1422_emailable_profit=web_email1422[(web_email1422.KGROUP==1) & (web_email1422.NUM_EMAILS>0)][['SCORE','NUM_EMAILS','ADJ_DIR_PROFIT']]
test1422_emailable_profit.shape
test1422_emailable_profit
web_email1422.columns
test1422_nonemail=web_email1422[(web_email1422.KGROUP==1) & (web_email1422.NUM_EMAILS==0)][['SCORE','NUM_EMAILS','ADJ_DIR_PROFIT']]
test1422_nonemail[web_email1422.SCORE>3.1202].ADJ_DIR_PROFIT.mean()
1.2111*13633+5.2047*43-2.3756*46+9961*0.6727
23683*0.9849
83*4.589+18*19.1682+0.7345*14403+1.1796*9179
test1422_nonemail[test1422_nonemail.NUM_EMAILS>0].HH_NUM.head().values
test1422_emailable_profit[test1422_emailable_profit.NUM_EMAILS>0].NUM_EMAILS.mean()
test1422_nonemailable_profit[test1422_nonemailable_profit.NUM_EMAILS>0].NUM_EMAILS.mean()
len(test1422_emailable_profit)
len(test1422_nonemail)
np.random.choice(10,5)
#tree for emailable
rnum=np.random.choice(len(test1422_emailable_profit),len(test1422_nonemail))
test_emailable_profit=test1422_emailable_profit.iloc[rnum,:]
test_emailable_profit.shape
X2=test_emailable_profit[['SCORE','NUM_EMAILS']]
X2.shape
y2=test_emailable_profit[['ADJ_DIR_PROFIT']]

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,train_size=0.7,random_state=1)
depth=10
result2=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree2=DecisionTreeRegressor(max_depth=i,min_samples_split=5,min_samples_leaf=5)
    tree2_model=tree2.fit(X2_train,y2_train)
    pred_train=tree2.predict(X2_train)
    pred_test=tree2.predict(X2_test)
    R_train=tree2.score(X2_train,y2_train)
    result2[i-2,1]=tree2.score(X2_train,y2_train)
    result2[i-2,2]=tree2.score(X2_test,y2_test)
    result2[i-2,0]=i
result2
tree_best_profit2=DecisionTreeRegressor(max_depth=2,min_samples_split=5,min_samples_leaf=5)
tree2_best_model_profit=tree_best_profit2.fit(X2,y2)
#train_score2=tree_best.predict_proba(X2_train)

tree_best_profit2.feature_importances_
gb_imp=pd.DataFrame({'variable':X2.columns,'importance':tree_best_profit2.feature_importances_}).sort_values(by='importance',ascending=False)
plt.barh(range(len(gb_imp['variable'])),gb_imp['importance'])
plt.yticks(range(len(gb_imp['variable'])), gb_imp['variable'])
for a,b in zip(gb_imp['importance'],range(len(gb_imp['variable']))): 
            plt.text(a+0.001, b, str(round(a,3)),fontsize=12)
plt.xlabel('Relative Importance')
plt.title('Top Important Variable of Tree')
plt.show()
dot_data = export_graphviz(tree2_best_model_profit, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("test_emailable_profit") 
dot_data = export_graphviz(tree2_best_model_profit, out_file=None, 
                         feature_names=list(test_emailable_profit.columns.values[:2]),  
                         class_names=np.array(['profit']),  
                         filled=True, rounded=True,
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph 
#tree for no email
X2n=test1422_nonemail[['SCORE']]
y2n=test1422_nonemail[['ADJ_DIR_PROFIT']]
X2n_train,X2n_test,y2n_train,y2n_test=train_test_split(X2n,y2n,train_size=0.7,random_state=1)
depth=20
result2n=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree2n=DecisionTreeRegressor(max_depth=i,min_samples_split=5,min_samples_leaf=5)
    tree2n_model=tree2n.fit(X2n_train,y2n_train)
    pred_train=tree2n.predict(X2n_train)
    pred_test=tree2n.predict(X2n_test)
    #R_train=tree2.score(X2n_train,y2n_train)
    result2n[i-2,1]=tree2n.score(X2n_train,y2n_train)
    result2n[i-2,2]=tree2n.score(X2n_test,y2n_test)
    result2n[i-2,0]=i
np.set_printoptions(threshold=50)
print(result2n[10:16])
tree_best_profit2n=DecisionTreeRegressor(max_depth=2,min_samples_split=5,min_samples_leaf=5)
tree2_best_model_profitn=tree_best_profit2n.fit(X2n,y2n)
#train_score2=tree_best.predict_proba(X2_train)
X2n_train.columns
y2n_train.columns
dot_datan = export_graphviz(tree2_best_model_profitn, out_file=None) 
graphn = graphviz.Source(dot_datan) 
graphn.render("test_noemail_profit") 
dot_datan = export_graphviz(tree2_best_model_profitn, out_file=None, 
                         feature_names=list(X2n.columns.values),  
                         class_names=np.array(['profit']),  
                         filled=True, rounded=True,
                         special_characters=True)  
graphn = graphviz.Source(dot_datan) 
graphn 
###

test1422=web_email1422[web_email1422.KGROUP==1][['SCORE','NUM_EMAILS','DIRECT_RESP','DIRECT_DMD','ADJ_DIR_PROFIT']]
test1422.NUM_EMAILS.max()
test1422
test1422.SCORE.max()
labels = [ "{0} - {1}".format(i, i + 1) for i in range(0,8) ]
labels
test1422['score_group']=pd.cut(test1422.SCORE,range(0,9),right=True,labels=labels)
test1422.head()
labels = [ "{0} - {1}".format(i, i + 5) for i in range(0,39,5) ]
labels
test1422['nemail_group']=pd.cut(test1422.NUM_EMAILS,range(0,41,5),right=False,labels=labels)
test1422[test1422.NUM_EMAILS==39].head()
resp=test1422.groupby(['score_group','nemail_group']).DIRECT_RESP.count()
resp.values
index1=resp.index.levels[0].values.categories.values
index2=resp.index.levels[1].values.categories.values
pd.crosstab(test1422.score_group,test1422.nemail_group,values=test1422.DIRECT_RESP,aggfunc='count')
print(pd.__version__)                      
!pip install --upgrade pandas                       
web_email_ex=web_email1422[web_email1422.NUM_EMAILS>0][['SCORE','NUM_EMAILS','DIRECT_RESP','DIRECT_DMD','ADJ_DIR_PROFIT']]
web_ex=web_email1422[web_email1422.NUM_EMAILS<=0][['SCORE','NUM_EMAILS','DIRECT_RESP','DIRECT_DMD','ADJ_DIR_PROFIT']]
web_email_ex.ADJ_DIR_PROFIT.mean()
web_ex.ADJ_DIR_PROFIT.mean()
web_email_ex.DIRECT_RESP.mean()
web_ex.DIRECT_RESP.mean()
web_email_stat1=web_email_ex.mean()
web_email_stat1['COUNT']=web_email_ex.SCORE.count()
web_email_stat1=web_email_stat1.astype(int)
web_email_stat1
web_email_stat=pd.concat([web_email_stat1,web_email_stat2])
web_ex_stat=web_ex.mean()
web_ex_stat['COUNT']=web_ex.SCORE.count()
web_ex_stat
sample=np.random.choice(len(web_email_ex),len(web_ex))
web_email_sample=web_email_ex.iloc[sample,:]
web_email_sample.mean()
web_email1422_modify[web_email1422_modify.KGROUP==1].count()
web_email1421_modify[web_email1421_modify.KGROUP==1].KGROUP.count()
web_email1421.shape
####
web_email1422_modify.head(3)
web_email1422_modify.iloc[:,[1,2,3,4,5,-1]].groupby(['KPANEL','NUM_EMAILS0_1']).mean()
web_email1422_modify.iloc[:,[1,2,3,4,5,-1]].groupby(['KPANEL']).mean()
web_email1422_modify.iloc[:,[2,3,4,5,-1]].groupby(['NUM_EMAILS0_1']).mean()
web_email1421_modify.shape
web_email1421_modify.iloc[:,[1,2,4,5,6,-1]].groupby(['KPANEL','NUM_EMAILS0_1']).mean()
web_email1421_modify.iloc[:,[1,2,4,5,6,-1]].groupby(['KPANEL']).mean()
web_email1421_modify.iloc[:,[2,4,5,6,-1]].groupby(['NUM_EMAILS0_1']).mean()

####Tree for 1421
test1421_emailable_profit=web_email1421[(web_email1421.KGROUP==1) & (web_email1421.NUM_EMAILS>0)][['SCORE','NUM_EMAILS','ADJ_DIR_PROFIT']]

test1421_nonemail=web_email1421[(web_email1421.KGROUP==1) & (web_email1421.NUM_EMAILS==0)][['SCORE','NUM_EMAILS','ADJ_DIR_PROFIT']]

len(test1421_emailable_profit)
len(test1421_nonemail)

#tree for emailable
rnum=np.random.choice(len(test1421_emailable_profit),len(test1421_nonemail))
test_emailable_profit=test1421_emailable_profit.iloc[rnum,:]
test_emailable_profit.shape
X2=test_emailable_profit[['SCORE','NUM_EMAILS']]
X2.shape
y2=test_emailable_profit[['ADJ_DIR_PROFIT']]

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,train_size=0.7,random_state=1)
depth=30
result2=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree2=DecisionTreeRegressor(max_depth=i,min_samples_split=5,min_samples_leaf=5)
    tree2_model=tree2.fit(X2_train,y2_train)
    pred_train=tree2.predict(X2_train)
    pred_test=tree2.predict(X2_test)
    R_train=tree2.score(X2_train,y2_train)
    result2[i-2,1]=tree2.score(X2_train,y2_train)
    result2[i-2,2]=tree2.score(X2_test,y2_test)
    result2[i-2,0]=i
result2
tree_best_profit2=DecisionTreeRegressor(max_depth=18,min_samples_split=5,min_samples_leaf=5)
tree2_best_model_profit=tree_best_profit2.fit(X2,y2)
#train_score2=tree_best.predict_proba(X2_train)

tree_best_profit2.feature_importances_
gb_imp=pd.DataFrame({'variable':X2.columns,'importance':tree_best_profit2.feature_importances_}).sort_values(by='importance',ascending=False)
plt.barh(range(len(gb_imp['variable'])),gb_imp['importance'])
plt.yticks(range(len(gb_imp['variable'])), gb_imp['variable'])
for a,b in zip(gb_imp['importance'],range(len(gb_imp['variable']))): 
            plt.text(a+0.001, b, str(round(a,3)),fontsize=12)
plt.xlabel('Relative Importance')
plt.title('Top Important Variable of Tree')
plt.show()
dot_data = export_graphviz(tree2_best_model_profit, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("test_emailable_profit") 
dot_data = export_graphviz(tree2_best_model_profit, out_file=None, 
                         feature_names=list(test_emailable_profit.columns.values[:2]),  
                         class_names=np.array(['profit']),  
                         filled=True, rounded=True,
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph 
#tree for no email
X2n=test1421_nonemail[['SCORE']]
y2n=test1421_nonemail[['ADJ_DIR_PROFIT']]
X2n_train,X2n_test,y2n_train,y2n_test=train_test_split(X2n,y2n,train_size=0.7,random_state=1)
depth=10
result2n=np.zeros((depth-1,3))
for i in range(2,depth+1):
    
    tree2n=DecisionTreeRegressor(max_depth=i,min_samples_split=5,min_samples_leaf=5)
    tree2n_model=tree2n.fit(X2n_train,y2n_train)
    pred_train=tree2n.predict(X2n_train)
    pred_test=tree2n.predict(X2n_test)
    #R_train=tree2.score(X2n_train,y2n_train)
    result2n[i-2,1]=tree2n.score(X2n_train,y2n_train)
    result2n[i-2,2]=tree2n.score(X2n_test,y2n_test)
    result2n[i-2,0]=i
result2n
tree_best_profit2n=DecisionTreeRegressor(max_depth=2,min_samples_split=5,min_samples_leaf=5)
tree2_best_model_profitn=tree_best_profit2n.fit(X2n,y2n)
#train_score2=tree_best.predict_proba(X2_train)
X2n_train.columns
y2n_train.columns
dot_datan = export_graphviz(tree2_best_model_profitn, out_file=None) 
graphn = graphviz.Source(dot_datan) 
graphn.render("test_noemail_profit") 
dot_datan = export_graphviz(tree2_best_model_profitn, out_file=None, 
                         feature_names=list(X2n.columns.values),  
                         class_names=np.array(['profit']),  
                         filled=True, rounded=True,
                         special_characters=True)  
graphn = graphviz.Source(dot_datan) 
graphn 
###