import numpy as np

CatName={}
CatName[1]='Vessel'
CatName[2]='V_Label'
CatName[3]='V_Cork'
CatName[4]='V_Parts_GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid_GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid_GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'

CatName[44]='bottle'
CatName[46]='wine glass'
CatName[47]='cup'
CatName[50]='spoon'
CatName[51]='bowl'
CatName[70]='toilet'
CatName[81]='sink'
CatName[86]='vase'



CatLiquid=6
CatSolid=10
CatFilled=16
CatVParts=4
SolidLabels={9,10,11,12,13}
LiquidLabels={6,7,9}
FilledLabels={6,7,8,9,10,11,12,13,15}
PartsLabels={2,3,4}

CatLossWeight={}
CatLossWeight['Vessel']=1
CatLossWeight['V_Label']=0.5
CatLossWeight['V_Cork']=0.5
CatLossWeight['V_Parts_GENERAL']=0.5
CatLossWeight['Ignore']=0
CatLossWeight['Liquid_GENERAL']=1
CatLossWeight['Liquid Suspension']=1
CatLossWeight['Foam']=1
CatLossWeight['Gel']=1
CatLossWeight['Solid_GENERAL']=1
CatLossWeight['Granular']=1
CatLossWeight['Powder']=1
CatLossWeight['Solid Bulk']=1
CatLossWeight['Vapor']=1
CatLossWeight['Other Material']=1
CatLossWeight['Filled']=1

CatLossWeight['bottle']=0.2
CatLossWeight['wine glass']=0.2
CatLossWeight['cup']=0.2
CatLossWeight['spoon']=0.2
CatLossWeight['bowl']=0.2
CatLossWeight['toilet']=0.1
CatLossWeight['sink']=0.1
CatLossWeight['vase']=0.1


CatNum={}
CatNum['Vessel']=-1
CatNum['V_Label']=-1
CatNum['V_Cork']=-1
CatNum['V_Parts_GENERAL']=-1
CatNum['Ignore']=-1
CatNum['Liquid_GENERAL']=-1
CatNum['Liquid Suspension']=-1
CatNum['Foam']=-1
CatNum['Gel']=-1
CatNum['Solid_GENERAL']=-1
CatNum['Granular']=-1
CatNum['Powder']=-1
CatNum['Solid Bulk']=-1
CatNum['Vapor']=-1
CatNum['Other Material']=-1
CatNum['Filled']=-1

CatNum['bottle']=-1
CatNum['wine glass']=-1
CatNum['cup']=-1
CatNum['spoon']=-1
CatNum['bowl']=-1
CatNum['toilet']=-1
CatNum['sink']=-1
CatNum['vase']=-1

def NormalizeWeight(SomeWeight=5,MaxWeight=20):
    SumExamples=0
    for nm in CatNum:
        if CatNum[nm] >= 0:
             SumExamples+=CatNum[nm]
    for nm in CatNum:
        if CatNum[nm] > 0:
             CatLossWeight[nm]*=np.min([(SumExamples/CatNum[nm]),MaxWeight])







