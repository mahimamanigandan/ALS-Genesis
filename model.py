#import all the required packages
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.target import ClassBalance
import imblearn
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df_ctrlvscase = pd.read_csv('/kaggle/input/end-als/end-als/transcriptomics-data/DESeq2/ctrl_vs_case.csv')

df_qc_staining = pd.read_csv('/kaggle/input/end-als/end-als/clinical-data/filtered-metadata/metadata/clinical/Answer ALS metadata datatable with QC Staining.csv')
df_qc_staining = df_qc_staining[['GUID','NEFH (SMI32)','ISL1','NKX6.1','TUBB3 (TuJ1)','s100b','Nestin']]
df_qc_staining = df_qc_staining.rename(columns = {'GUID':'Participant_ID'})

df_geno_bin = pd.read_csv('/kaggle/input/end-als/end-als/genomics-data/geno_bin.csv',index_col = [0]).T
df_geno_bin.reset_index(inplace=True)
df_geno_bin = df_geno_bin.rename(columns = {'index':'unformatted_ID'})
df_geno_bin['Participant_ID'] = df_geno_bin.unformatted_ID.apply(lambda x: x[5:])
df_geno_bin = df_geno_bin.drop(columns=['unformatted_ID'])
df_combined1 = pd.merge(df_ctrlvscase, df_qc_staining, on='Participant_ID', how='inner')
df_combined1.to_csv('/kaggle/working/stain_rna.csv')

#dataset for random forest classifier
df_combined = pd.merge(df_combined1, df_geno_bin, on='Participant_ID')
df_combined.to_csv('/kaggle/working/stain_genomics_rna.csv')

#dataset for gwas
df_label = df_ctrlvscase[['Participant_ID','CtrlVsCase_Classifier']]
df_genome_wide = pd.merge(df_label, df_geno_bin, on='Participant_ID', how='inner')

#function for Random forest classifier
def Random_Forest_Classifier(dataset,do_SMOTE,feature_idx,label_idx,ntrees):
    df = dataset.fillna(0)
    
    X, y = df.iloc[:,feature_idx:],df.iloc[:,label_idx]

    #standardize the dataset for ML
    X_standard = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X_standard, columns=X.columns)

    #Check the class balance
    graph = ClassBalance(labels=["class 0","class 1"])
    graph.fit(y)        
    graph.show()
    
    if do_SMOTE == True:
        oversample = imblearn.over_sampling.SMOTE()

        # fit and apply the transform
        X_over, y_over = oversample.fit_resample(X, y)
    else:
        X_over, y_over = X, y
        
    print('oversampled shape of X is ',X_over.shape)
    print('oversampled shape of y is ',y_over.shape,'\n')
    features = X_over.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=0)  

    #define a RF classifier object and fit a model
    Random_Forest = RandomForestClassifier(n_estimators=ntrees,random_state=123)
    Random_Forest.fit(X_train,y_train)

    #rank the features and store in a dataframe
    feature_importance_values = Random_Forest.feature_importances_
    feature_importances = pd.DataFrame({'feature': features, 'importance':feature_importance_values})
    key_features = feature_importances[feature_importances.importance != 0.0]
    key_features = key_features.sort_values(by='importance',ascending=False)
    key_features.to_csv('/kaggle/working/key_features_dna_rna.csv')
    key_col_names = key_features["feature"].tolist()
    newXtrain = X_train[X_train.columns.intersection(key_col_names)]
    newXtest = X_test[X_test.columns.intersection(key_col_names)]

    print('reduced dimension of training input from ', X_train.shape, 'to ', newXtrain.shape, 'by dropping irrelevant features \n')

    #ploting the top key feature importance
    key_features.head(20).plot(kind='barh',x='feature',y='importance',color='blue')
    plt.show()
    
    #train the model
    Random_Forest.fit(newXtrain,y_train)
    pred_temp = Random_Forest.predict(newXtest)
    pred_final = [round(value) for value in pred_temp]
    print('Check that the output prediction is not simply returning a trivial solution of all 1s \n',pred_final,'\n')
    
    accuracy = accuracy_score(y_test, pred_final)
    print("Simple accuracy is: %.2f%%" % (accuracy * 100.0),'\n')
    print('ROC AUC is ', roc_auc_score(y_test, pred_final),'\n')

    explainer = shap.TreeExplainer(Random_Forest)
    shap_values = explainer.shap_values(newXtrain)
    shap.summary_plot(shap_values[1], newXtrain)

#function for genome wide association study
def pseudo_genome_wide(dataset,label_name,feature_idx):
    features_list=dataset.columns.tolist()
    features_list = features_list[feature_idx:]
    print('checking ',len(features_list), ' SNP variations and comparing between classes\n')
    count=0
    case_lst=[]
    ctrl_lst=[]

    #calculate sample size
    total_patients = dataset.shape[0]
    case_patients = dataset.loc[dataset[label_name]>0].shape[0]
    ctrl_patients = total_patients - case_patients

    #fillna
    dataset = dataset.fillna(0)
    for feature in features_list:
        var_sum = dataset.groupby(by=[label_name])[feature].sum()
        ctrl_sum = var_sum.iloc[[0]]
        case_sum = var_sum.iloc[[1]]
        case_sum = float(case_sum.values)
        ctrl_sum = float(ctrl_sum.values)
        case_pct = (case_sum / case_patients)*100
        ctrl_pct = (ctrl_sum / ctrl_patients)*100

        case_lst.append(case_pct)
        ctrl_lst.append(ctrl_pct)
        
        #print progress
        count = count+1
        if count%10000 == 0:
            print('checked ', count, ' SNPs so far')
    #make a new dataframe for plotting and analysis
    d = {'case_pct': case_lst, 'ctrl_pct': ctrl_lst}
    df_variants = pd.DataFrame(data=d, index=features_list)
    df_variants['delta'] = df_variants['case_pct'] - df_variants['ctrl_pct']
    df_variants = df_variants.sort_values(by='delta',ascending=False)

    AX = df_variants.head(10).plot.bar(rot=0,figsize=(20,20))
    AX.set_xlabel("key variant identified by ML")
    AX.set_ylabel("Percent of participants with that variant")
    df_variants.to_csv('/kaggle/working/gwas.csv')
    
def genome_wide_rna(dataset,label_name,feature_idx):
    features_list=dataset.columns.tolist()
    features_list = features_list[feature_idx:]
    print('checking ',len(features_list), ' gene expression levels and comparing between classes\n')
    count=0
    case_lst=[]
    ctrl_lst=[]

    #calculate sample size
    total_patients = dataset.shape[0]
    case_patients = dataset.loc[dataset[label_name]>0].shape[0]
    ctrl_patients = total_patients - case_patients
    
    #normalize data 
    df_genome_wide_rna = dataset
    df_genome_wide_rna = df_genome_wide_rna.fillna(0)
    df_genome_wide_rna = df_genome_wide_rna.drop(columns=['Participant_ID'])
    x_scaled = MinMaxScaler().fit_transform(df_genome_wide_rna)
    df_genome_wide_rna = pd.DataFrame(x_scaled, columns=df_genome_wide_rna.columns)
    for feature in features_list:
        #sum gene expression for each patient group for each feature
        var_sum = df_genome_wide_rna.groupby(by=[label_name])[feature].sum()
        ctrl_sum = var_sum.iloc[[0]]
        case_sum = var_sum.iloc[[1]]
        case_sum = float(case_sum.values)
        ctrl_sum = float(ctrl_sum.values)
        case_pct = (case_sum / case_patients)
        ctrl_pct = (ctrl_sum / ctrl_patients)
        case_lst.append(case_pct)
        ctrl_lst.append(ctrl_pct)
        
        #track progress
        count = count+1
        if count%10000 == 0:
            print('checked ', count, ' genes so far')
    #make a new dataframe for plotting and analysis
    d = {'case_pct': case_lst, 'ctrl_pct': ctrl_lst}
    df_variants = pd.DataFrame(data=d, index=features_list)
    df_variants['delta'] = df_variants['case_pct'] - df_variants['ctrl_pct']
    df_variants = df_variants.sort_values(by='delta',ascending=False)

    AX = df_variants.head(10).plot.bar(rot=0,figsize=(20,20))
    AX.set_xlabel("Gene")
    AX.set_ylabel("Average relative expression of genes for top 10 differences between classes")
    df_no_expression = df_variants[(df_variants['case_pct']==0)&(df_variants['ctrl_pct']>0.07)]
    AX = df_no_expression.head(10).plot.bar(rot=0,figsize=(15,15))
    AX.set_xlabel("Gene")
    AX.set_ylabel("Average relative expression for genes not expressed by class = 1 but expressed by class =0")
    
#run the random forest classifier
Random_Forest_Classifier(dataset=df_combined,do_SMOTE=True,feature_idx=2,label_idx=1,ntrees=1000)

#run the pseudo-gwas
pseudo_genome_wide(dataset=df_genome_wide,label_name = 'CtrlVsCase_Classifier',feature_idx = 2)

#run the rna comparison function
genome_wide_rna(dataset=df_ctrlvscase,label_name = 'CtrlVsCase_Classifier',feature_idx = 2)


#run the random forest classifier
df_BulbarvsLimb = pd.read_csv('/kaggle/input/end-als/end-als/transcriptomics-data/DESeq2/bulbar_vs_limb.csv')
df_BulbarvsLimb_temp = pd.merge(df_BulbarvsLimb,df_geno_bin,on='Participant_ID',how='inner')
Random_Forest_Classifier(dataset=df_BulbarvsLimb_temp,do_SMOTE=True,feature_idx=2,label_idx=1,ntrees=1000)

#run the pseudo-gwas
df_label = df_BulbarvsLimb[['Participant_ID','SiteOnset_Class']]
df_genome_wide = pd.merge(df_label, df_geno_bin, on='Participant_ID', how='inner')
pseudo_genome_wide(dataset=df_genome_wide,label_name = 'SiteOnset_Class',feature_idx = 2)

#run the rna comparison function
genome_wide_rna(dataset=df_BulbarvsLimb,label_name = 'SiteOnset_Class',feature_idx = 2)

df_alsfrs_r = pd.read_csv("/kaggle/input/end-als/end-als/clinical-data/filtered-metadata/metadata/clinical/ALSFRS_R.csv")
df_alsfrs_r['functional_sum']  = df_alsfrs_r['alsfrs1']+df_alsfrs_r['alsfrs2']+df_alsfrs_r['alsfrs3']+df_alsfrs_r['alsfrs4']+df_alsfrs_r['alsfrs5']+df_alsfrs_r['alsfrs6']+df_alsfrs_r['alsfrs7']+df_alsfrs_r['alsfrs8']+df_alsfrs_r['alsfrs9']

#calculate the maximum frs score change 
df_alsfrs_r_summary = df_alsfrs_r[['SubjectUID','Visit_Date','functional_sum']]
df_min = df_alsfrs_r_summary.groupby(by=['SubjectUID'],as_index=False)['functional_sum'].min()
df_min_date = df_alsfrs_r_summary.merge(df_min, on=['SubjectUID','functional_sum'],how='inner')
df_min_date['min_functional_sum'] = df_min_date['functional_sum']
df_min_date = df_min_date.drop(columns='functional_sum')
df_initial = df_alsfrs_r_summary[df_alsfrs_r_summary['Visit_Date']==0]
df_initial['initial_functional_sum']  = df_initial['functional_sum']
df_initial =df_initial.drop(columns = ['functional_sum','Visit_Date'])

df_alsfrs_r_data = pd.merge(df_initial, df_min_date, on='SubjectUID',how='inner')

df_alsfrs_r_data  = df_alsfrs_r_data.drop_duplicates(['SubjectUID', 'min_functional_sum'])
df_alsfrs_r_data['delta_functional_score'] = df_alsfrs_r_data['initial_functional_sum'] - df_alsfrs_r_data['min_functional_sum']

#calculate a score for rapid frs changes
df_alsfrs_r_data['rapid_onset_score'] = df_alsfrs_r_data['delta_functional_score'] / df_alsfrs_r_data['Visit_Date']

#plot the data
df_alsfrs_r_data.hist(column='rapid_onset_score',bins=30)
df_alsfrs_r_data.plot(kind='scatter',x='Visit_Date',y='delta_functional_score')
print('mean rapid onset score is ', np.mean(df_alsfrs_r_data['rapid_onset_score']))
df_alsfrs_r_data['rapid_onset_class'] = np.where(df_alsfrs_r_data['rapid_onset_score'] >= np.mean(df_alsfrs_r_data['rapid_onset_score']), 1, 0)
df_alsfrs_r_data = df_alsfrs_r_data.rename(columns = {'SubjectUID':'Participant_ID'})

df_alsfrs_r_data = df_alsfrs_r_data.rename(columns = {'SubjectUID':'Participant_ID'})
df_rna_onset = pd.merge(df_alsfrs_r_data,df_ctrlvscase, on='Participant_ID',how='inner')

df_alsfrs_r_ltsurvival = df_alsfrs_r_data[(df_alsfrs_r_data['delta_functional_score']<5)&(df_alsfrs_r_data['Visit_Date']>(365*2))]
df_alsfrs_r_ltsurvival['survival_prospect'] = 1

df_alsfrs_r_stsurvival = df_alsfrs_r_data[(df_alsfrs_r_data['delta_functional_score']>10)&(df_alsfrs_r_data['Visit_Date']<(365*1))]
df_alsfrs_r_stsurvival['survival_prospect'] = 0

df_survival = pd.concat([df_alsfrs_r_ltsurvival,df_alsfrs_r_stsurvival])
df_survival = df_survival.rename(columns = {'SubjectUID':'Participant_ID'})

#merge with transcriptomics data
df_survival_rna = pd.merge(df_survival,df_ctrlvscase, on='Participant_ID',how='inner')

Random_Forest_Classifier(dataset=df_rna_onset,do_SMOTE=True,feature_idx=8,label_idx=6,ntrees=1000)

from sklearn.ensemble import ExtraTreesClassifier

# define dataset
X, y = df_survival_rna.iloc[:,8:],df_survival_rna.iloc[:,6]

#save the feature names for later
feature_names = X.columns.tolist()

#https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets
TOP_FEATURES = 5

forest = ExtraTreesClassifier(n_estimators=1000, max_depth=5, random_state=1)
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]
indices = indices[:TOP_FEATURES]

print('Top features:')
for f in range(TOP_FEATURES):
    print(f + 1, feature_names[indices[f]], importances[indices[f]])
    
def plot_feature_importances(df):
    #Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    #Normalise the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    #Make a horizontal bar graph of feature importances
    plt.figure(figsize = (10,6))
    ax = plt.subplot()
    
    ax.barh(list(reversed(list(df.index[:15]))),
           df['importance_normalized'].head(15),
           align = 'center', edgecolor = 'k')
    
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    #Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importance')
    plt.show()
    return df

demographics = pd.read_csv('/kaggle/input/end-als/end-als/clinical-data/filtered-metadata/metadata/clinical/Demographics.csv')
alsfrs_scores = pd.read_csv('/kaggle/input/end-als/end-als/clinical-data/filtered-metadata/metadata/clinical/ALSFRS_R.csv')
bulbar_vs_limb = pd.read_csv('/kaggle/input/end-als/end-als/transcriptomics-data/DESeq2/bulbar_vs_limb.csv')
ctrl_vs_case = pd.read_csv('/kaggle/input/end-als/end-als/transcriptomics-data/DESeq2/ctrl_vs_case.csv')

import seaborn as sns
sns.histplot(demographics.age);
sns.histplot(alsfrs_scores.alsfrsdt);

rows = 1000 # specify 'None' if want to read whole file
df_genes = pd.read_csv('../input/cusersmarildownloadsgenescsv/genes.csv', delimiter=';', encoding = "ISO-8859-2", nrows = rows)
df_genes.dataframeName = 'genes.csv'
nRow, nCol = df_genes.shape
print(f'There are {nRow} rows and {nCol} columns')
df_genes.head()

df_genes["Gene"].value_counts()

df_genes["Associated_ND"].value_counts()

df_genes["Associated_ND"].value_counts().plot.bar(color=['orange', 'grey','purple','pink'], title='ALS Associated ND Genes');

df_genes["Phenotype_influence"].value_counts()

df_genes["Phenotype_influence"].value_counts().plot.bar(color=['purple', 'red','grey','pink', 'Chartreuse', 'Coral', 'DarkOrchid', 'black'], title='ALS Associated ND Genes');

long_survival = df_genes[(df_genes['Phenotype_influence']=='Longer survival')]
long_survival.head()

short_survival = df_genes[(df_genes['Phenotype_influence']=='Shorter survival')].reset_index(drop=True)
short_survival.head()

phenotype = df_genes[(df_genes['Phenotype_influence']=='Limb-onset, early age of onset and shorter survival')].reset_index(drop=True)
phenotype.head()

df_protein = '/kaggle/input/pro-bio/Protein-Biogrid.csv'
df = pd.read_csv(df_protein)
df.head()

df_genes_ALSoD = pd.read_csv('/kaggle/input/proteinproteininteraction/genes_from_alsod.csv')
genes_ALS = list( df_genes_ALSoD['Gene symbol'])

A = df['Official Symbol Interactor A'].isin(genes_ALS) & (df['Organism Name Interactor A'] == 'Homo sapiens') & (df['Organism Name Interactor B'] == 'Homo sapiens')
B = df['Official Symbol Interactor B'].isin(genes_ALS) & (df['Organism Name Interactor A'] == 'Homo sapiens') & (df['Organism Name Interactor B'] == 'Homo sapiens')
data1 =  df[A]['Official Symbol Interactor B'].value_counts() 
data2 =  df[B]['Official Symbol Interactor A'].value_counts() 
d = data1.to_frame().join(data2, how = 'outer')
dataset_protein = d.fillna(0)
dataset_protein['Count All Interactions'] = dataset_protein.iloc[:,0] + dataset_protein.iloc[:,1]
d = dataset_protein.join( df_genes_ALSoD.set_index('Gene symbol') )
d = d.sort_values('Count All Interactions',ascending = False)
d = d[['Count All Interactions', 'Gene name', 'Category' , 'Official Symbol Interactor A', 'Official Symbol Interactor B'] ]
d.columns = ['Count All BIOGRID Interactions', 'Gene name', 'Relation to ALS' , 'Count Left Interactions', 'Count Right Interactions']  

print(d.columns)
d.index.name = 'Gene symbol'
d.dropna(axis=0,inplace=True)
d

from scipy import stats
df_BulbarvsLimb = pd.read_csv('/kaggle/input/end-als/end-als/transcriptomics-data/DESeq2/bulbar_vs_limb.csv')
df_BulbarvsLimb.boxplot(column=['SOD1'],by='SiteOnset_Class')

groupA = df_BulbarvsLimb.where(df_BulbarvsLimb.SiteOnset_Class== 0).dropna()['SOD1']
groupB = df_BulbarvsLimb.where(df_BulbarvsLimb.SiteOnset_Class== 1).dropna()['SOD1']

print('Lower SOD1 expression levels were found for limb onset with p value ', stats.ttest_ind(groupA,groupB)[1])

dict_characters = {0: 'Bulbar', 1: 'Limb'} # double check this
print(dict_characters)
sns.set_style("darkgrid")
plt = sns.FacetGrid(bulbar_vs_limb, hue='SiteOnset_Class',aspect=3)
plt.map(sns.kdeplot,'SOD1',shade=False)
plt.set(xlim=(bulbar_vs_limb['SOD1'].min(), bulbar_vs_limb['SOD1'].max()))
plt.add_legend()
plt.set_axis_labels('SOD1', 'Proportion')
plt.fig.suptitle('SOD1 counts vs Diagnosis (0 = Bulbar; 1 = Limb)')

