'''
Created on 29 Oct 2017

@author: Team 3
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

statecourtsdata = "/Users/bryan\Downloads/Telegram Desktop/New folder/Team3_data_StateCourts.csv"

#dataframe for descriptive statistics
df = pd.read_csv(statecourtsdata)

#function for printing BASIC X-VARIABLES
def condition(i):
    e = ['1','2','3','4','5','6']
    if i in e:
        return i
    else:
        return '7'
df['Category_of_Claim'] = df['Category_of_Claim'].apply(condition)

def QualitativeDescribe(i):
    print i + ":"
    print df[i].value_counts().to_csv(sep='\t') 
    print "---------"
    
def QualitativeBar(i):
    print df[i].value_counts().plot(kind='bar')
    print plt.title(i)
    plt.show()
    
def QuantitativeHist(i,x):
    plt.hist(df[i], bins = x)
    plt.title("Distribution of %s" %(i))
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.show()

print "DESCRIPTION OF X-VARIABLES"
QualitativeDescribe('Category_of_Claim')
QualitativeDescribe('Claimant_law_firm_size')
QualitativeDescribe('Defendant_law_firm_size')
QualitativeDescribe('Claimant_Type')
QualitativeDescribe('Defendant_Type')
QualitativeDescribe('Coram')
QualitativeBar('Coram')

#getting rid of plaintiff and defendants in person
def condition2(i):
    e = [-1]
    if i in e:
        return None
    else:
        return i
    
df['Claimant_PQE_no_LIP'] = df['Claimant_PQE'].apply(condition2)
df['Defendant_PQE_no_LIP'] = df['Defendant_PQE'].apply(condition2)
print df[['Coram_Experience','Defendant_PQE_no_LIP','Claimant_PQE_no_LIP']].describe()

#BASIC Y-VARIABLES
print "DESCRIPTION OF Y-VARIABLES"
print df[['Issues','Duration_of_Case','Stripped_Word_Count']].describe()
print "---------"
QualitativeDescribe('Success')
QuantitativeHist('Stripped_Word_Count',10)
QuantitativeHist('Issues',7)
QuantitativeHist('Duration_of_Case',7)

#CROSSTABS
def CrossTab(x,y):
    print x + " against " + y
    print pd.crosstab(df[x] , df[y])
    print "---------"

def CrossTab2(x,y,z):
    print x + " against " + y + " and " +z
    print pd.crosstab(df[x] , [df[y],df[z]])
    print "---------"

#Number of each cat per year
Cat_Year = CrossTab('Category_of_Claim', 'Year_of_Decision')
#Number of Cases handled by each Size per year
SizeC_Year = CrossTab('Claimant_law_firm_size', 'Year_of_Decision')
SizeD_Year = CrossTab('Defendant_law_firm_size', 'Year_of_Decision')
#Number of each type of Party per year
TypeC_Year = CrossTab('Claimant_Type', 'Year_of_Decision')
TypeD_Year = CrossTab('Defendant_Type', 'Year_of_Decision')
#Number of successful claims per year
Success_Year = CrossTab('Success', 'Year_of_Decision')
#Number of cases the Coram took per year
Coram_Year = CrossTab('Coram','Year_of_Decision')
#Number of party Types per Cat
TypeC_Cat = CrossTab('Claimant_Type', 'Category_of_Claim')
TypeD_Cat = CrossTab('Defendant_Type', 'Category_of_Claim')
#Number of Cases handled by each Size per category
SizeC_Cat = CrossTab('Claimant_law_firm_size', 'Category_of_Claim')
SizeD_Cat = CrossTab('Defendant_law_firm_size', 'Category_of_Claim')
#Number of Party Type per firm Size
SizeC_TypeC = CrossTab('Claimant_law_firm_size', 'Claimant_Type')
SizeD_TypeD = CrossTab('Defendant_law_firm_size', 'Defendant_Type')
#Cat_Coram = CrossTab('CategoryofClaim','Coram')

#Number of Success per firm Size
SizeC_TypeC = CrossTab('Claimant_law_firm_size', 'Success')
SizeD_TypeD = CrossTab('Defendant_law_firm_size', 'Success')

#Number of Success for each matchup
CrossTab2('Claimant_law_firm_size','Success','Defendant_law_firm_size')
#Number of matchups per category
CrossTab2('Claimant_law_firm_size','Category_of_Claim','Defendant_law_firm_size')
#DESCRIBES
def GroupDescribe(x,y):
    print 'Average '+ x +" per "+y
    print df[x].groupby(df[y]).describe()
    print "---------"

#YEAR
#Average Stripped Per Year
Stripped_Year = GroupDescribe('Stripped_Word_Count', 'Year_of_Decision')
#Average Issues per Year
Issues_Year = GroupDescribe('Issues', 'Year_of_Decision')
#Average experience of the Coram each year
Exp_Year = GroupDescribe('Coram_Experience', 'Year_of_Decision')
#Average ResolutionTime per Year
Time_Year = GroupDescribe('Duration_of_Case', 'Year_of_Decision')
#Average PQE per Year
PQEC_Year = GroupDescribe('Claimant_PQE_no_LIP', 'Year_of_Decision')
PQED_Year = GroupDescribe('Defendant_PQE_no_LIP', 'Year_of_Decision')

#TIME
#Average time taken per category
Time_Category = GroupDescribe('Duration_of_Case', 'Category_of_Claim')
#Average time taken to Adjudicate per year of experience
Time_Exp = GroupDescribe('Duration_of_Case','Coram_Experience')
#Average time taken per Size
Time_SizeC = GroupDescribe('Duration_of_Case','Claimant_law_firm_size')
Time_SizeD = GroupDescribe('Duration_of_Case','Defendant_law_firm_size')
#Average time taken per Issue
Time_Issue =  GroupDescribe('Duration_of_Case','Issues')
#Average time taken per success
Time_Success =  GroupDescribe('Duration_of_Case','Success')
#Average time taken per party Type 
Time_TypeC = GroupDescribe('Duration_of_Case','Claimant_Type')
Time_TypeD = GroupDescribe('Duration_of_Case','Defendant_Type')

#CAT
#Average Issues Per Category
Issues_Cat = GroupDescribe('Issues', 'Category_of_Claim')
#Average PQE per Category
PQEC_Cat = GroupDescribe('Claimant_PQE_no_LIP', 'Category_of_Claim')
PQED_Cat = GroupDescribe('Defendant_PQE_no_LIP', 'Category_of_Claim')
#Average Stripped per Category
Stripped_Cat = GroupDescribe('Stripped_Word_Count', 'Category_of_Claim')

#CORAM
#Average Stats per Coram
Stripped_Coram = GroupDescribe('Stripped_Word_Count', 'Coram')
Time_Coram = GroupDescribe('Duration_of_Case', 'Coram')
Issues_Coram = GroupDescribe('Issues', 'Coram')

#CORAM EXP
#Average Stats per Year of Coram Exp
Stripped_Exp = GroupDescribe('Stripped_Word_Count', 'Coram_Experience')
Time_Exp = GroupDescribe('Duration_of_Case', 'Coram_Experience')
Issues_Exp = GroupDescribe('Issues', 'Coram_Experience')

#Average PQE per firm size
PQEC_Size = GroupDescribe('Claimant_PQE_no_LIP', 'Claimant_law_firm_size')
PQED_Size = GroupDescribe('Defendant_PQE_no_LIP', 'Defendant_law_firm_size')
#Average Stripped per firm size
PQEC_Stripped = GroupDescribe('Stripped_Word_Count', 'Claimant_law_firm_size')
PQED_Stripped = GroupDescribe('Stripped_Word_Count', 'Defendant_law_firm_size')
#Average Issues per firm size
PQEC_Issues = GroupDescribe('Issues', 'Claimant_law_firm_size')
PQED_Issues = GroupDescribe('Issues', 'Defendant_law_firm_size')

#Average experience of the Coram per category
Exp_Cat = GroupDescribe('Coram_Experience','Category_of_Claim')

#SCATTER PLOTS
def ScatterPlotwLine(a,b):
    fit=np.polyfit(df[a],df[b],1)
    fit_fn=np.poly1d(fit)
    plt.scatter(df[a],df[b])
    plt.plot(df[a],df[b],'ro',df[a],fit_fn(df[a]),'b')
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title('Scatter plot of %s against %s' % (a,b))
    plt.show()

def ScatterPlot(a,b):
    plt.scatter(df[a],df[b])
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title('Scatter plot of %s against %s' % (a,b))
    plt.show()
    
df['Category_of_Claim'] = df['Category_of_Claim'].astype("int")

ScatterPlot('Claimant_PQE_no_LIP','Issues')
ScatterPlot('Claimant_PQE_no_LIP','Stripped_Word_Count')
ScatterPlot('Claimant_PQE_no_LIP','Duration_of_Case')

ScatterPlot('Defendant_PQE_no_LIP','Issues')
ScatterPlot('Defendant_PQE_no_LIP','Stripped_Word_Count')
ScatterPlot('Defendant_PQE_no_LIP','Duration_of_Case')

ScatterPlotwLine('Issues','Stripped_Word_Count')
ScatterPlotwLine('Duration_of_Case','Stripped_Word_Count')
ScatterPlotwLine('Issues','Duration_of_Case')

ScatterPlot('Claimant_PQE_no_LIP','Defendant_PQE_no_LIP')
ScatterPlotwLine('Coram_Experience','Duration_of_Case')

#dataframe for analytical statistics
dfAnalysis=pd.read_csv(statecourtsdata)

#recalibrating categories
dfAnalysis['Claimant_Type']=dfAnalysis['Claimant_Type'].astype('category')
dfAnalysis['Defendant_Type']=dfAnalysis['Defendant_Type'].astype('category')
dfAnalysis['Year_of_Decision']=dfAnalysis['Year_of_Decision'].astype('category')

# Transforming CategoryofClaim to dummy variables
dfAnalysis=dfAnalysis.join(dfAnalysis['Category_of_Claim'].str.get_dummies(','))
dfAnalysis=dfAnalysis.rename(columns={'1':"Contract" ,'2':"Tort" , '3':"Property" , '4':"CivPro" , '5':"MiscStat" , '6':"MiscNonStat"})
# df['CategoryofClaim']=df['CategoryofClaim'].astype('category')

#recalibrating word count
dfAnalysis['Raw']=np.log(dfAnalysis['Raw'])
dfAnalysis['Stripped_Word_Count']=np.log(dfAnalysis['Stripped_Word_Count'])
dfAnalysis['Difference']=np.log(dfAnalysis['Difference'])

#cleaning data by dropping NaN and strings in the data
dfCleaned=dfAnalysis.dropna()
#dropping plaintiff and defendants in person
dfCleaned1=dfCleaned[dfCleaned.Claimant_law_firm_size!=0]
dfCleaned1=dfCleaned1[dfCleaned.Defendant_law_firm_size!=0]
#change lawfirms to categorical variables
dfCleaned['Claimant_law_firm_size']=dfCleaned['Claimant_law_firm_size'].astype('category')
dfCleaned['Defendant_law_firm_size']=dfCleaned['Defendant_law_firm_size'].astype('category')
dfCleaned1['Claimant_law_firm_size']=dfCleaned1['Claimant_law_firm_size'].astype('category')
dfCleaned1['Defendant_law_firm_size']=dfCleaned1['Defendant_law_firm_size'].astype('category')

#dataframe with top10 corams for cases heard
dfCleaned2 = dfCleaned1[dfCleaned1["Coram"].isin(["Koh Juay Kherng", "Tan May Tee", "Loo Ngan Chor", "Leslie Chew", "Chiah Kok Khun", "Seah Chi Ling"])]

#running the linear and logistic models
#we have dropped contract from the dummy variables such that the other categories of claims are measures as against contract

#running the initial regression model
linearmodel_0=smf.ols(formula = 'Stripped_Word_Count ~  Tort + Property + CivPro + MiscStat + MiscNonStat + Claimant_Type + Defendant_Type + Duration_of_Case + Coram_Experience + Issues + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE', data=dfCleaned1).fit()
#running the linear regression model with corams as categorical values
linearmodel_1=smf.ols(formula = 'Stripped_Word_Count ~  Tort + Property + CivPro + MiscStat + MiscNonStat + Claimant_Type + Defendant_Type + Duration_of_Case + Coram_Experience + Issues + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE + C(Coram)', data=dfCleaned1).fit()
#running the linear regression model for duration of case against independent variables
linearmodel_2=smf.ols(formula = 'Duration_of_Case ~  Tort + Property + CivPro + MiscStat + MiscNonStat + Claimant_Type + Defendant_Type + Issues +  + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE + Coram_Experience', data=dfCleaned1).fit()
#running the linear regression model with top 10 corams as categorical values
linearmodel_3=smf.ols(formula = 'Stripped_Word_Count ~  Tort + Property + CivPro + MiscStat + MiscNonStat + Claimant_Type + Defendant_Type + Duration_of_Case + Coram_Experience + Issues + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE + C(Coram)', data=dfCleaned2).fit()

#printing results
print "Model to illustrate Relationship between Independent Variables and Stripped Word Count"
print linearmodel_0.summary()
print "---------"
print "Model to illustrate Relationship between Independent Variables with Coram and Stripped Word Count"
print linearmodel_1.summary()
print "---------"
print "Model to illustrate Relationship between Independent Variables and Duration of Case"
print linearmodel_2.summary()
print "---------"
print "Model to illustrate Relationship between Independent Variables and Stripped Word Count (Coram with 10 or more cases)"
print linearmodel_3.summary()
print "---------"

# running the logit model
#logit model of likelihood of success against independent variables
logitmodel_0=smf.logit(formula = 'Success ~ Claimant_Type + Defendant_Type + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE', data=dfCleaned1).fit()
#logit model of likelhood of success against independent variables with pf/df in person
logitmodel_1=smf.logit(formula = 'Success ~ Claimant_Type + Defendant_Type + Claimant_law_firm_size + Defendant_law_firm_size + Claimant_PQE + Defendant_PQE', data=dfCleaned).fit()

#defining functions for statistical cases and measures         
def StatisticalMeasures(predictedY,actualY):
    truepositives = 0
    truenegatives = 0
    falsepositives = 0
    falsenegatives = 0
    for i in range(len(predictedY)):
        if predictedY[i] == actualY[i] ==1:
            truepositives += 1
        elif predictedY[i]==1 and predictedY[i]!=actualY[i]:
            falsepositives +=1
        elif predictedY[i] == actualY[i] ==0:
            truenegatives+=1
        else:
            falsenegatives+=1
    return [truepositives,truenegatives,falsepositives,falsenegatives]

def Accuracy(predictedY,actualY):
    accuracyValue=(StatisticalMeasures(predictedY, actualY)[0]+StatisticalMeasures(predictedY, actualY)[1])/(len(predictedY)+0.0)
    return accuracyValue
def Precision(predictedY,actualY):
    precisionValue=(StatisticalMeasures(predictedY, actualY)[0]+0.0)/(StatisticalMeasures(predictedY, actualY)[0]+StatisticalMeasures(predictedY, actualY)[2])
    return precisionValue
def Recall(predictedY,actualY):
    recallValue=(StatisticalMeasures(predictedY, actualY)[0]+0.0)/((StatisticalMeasures(predictedY, actualY)[0]+(StatisticalMeasures(predictedY, actualY)[3]+0.0)))
    return recallValue                          

# determine actual y and predicted y
actualy = dfCleaned['Success'].tolist()
  
casedependentV = pd.concat([dfCleaned1['Claimant_law_firm_size'], dfCleaned1['Defendant_law_firm_size'], \
                              dfCleaned1['Claimant_Type'], dfCleaned1['Defendant_Type'], \
                              dfCleaned1['Claimant_PQE'], dfCleaned1['Defendant_PQE']], axis = 1)
predictedy_float = logitmodel_0.predict(casedependentV).tolist()
predictedy_round = map(round, predictedy_float)
   
# determine statistical measures and print
precisionscore = Precision(predictedy_round, actualy)
recallscore = Recall(predictedy_round, actualy)
accuracyscore = Accuracy(predictedy_round, actualy)
F1score = 2 * (precisionscore * recallscore) / (precisionscore + recallscore)
  
# print logitmodel and summaries
#logit model without litigants in person
print "Model to illustrate Relationship between Independent Variables and likelihood of success of claim"
print logitmodel_0.summary()
print "---------"
print "Accuracy: ", accuracyscore
print "Precision: ", precisionscore
print "Recall: ", recallscore
print "F1: ", F1score
print "---------"

#logit model with litigants in person
print "Model to illustrate Relationship between Independent Variables and likelihood of success of claim"
print logitmodel_1.summary()
