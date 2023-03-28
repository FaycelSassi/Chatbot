
import pymongo
import pandas as pd
# retrieve all documents from the collection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["rasa"]
collection = db["chap2"]
data = list(collection.find())

# create a Pandas DataFrame from the list of dictionaries
Xdf = pd.DataFrame(data)

Xdf = Xdf.drop('_id', axis=1)
paramlist = Xdf.values.tolist()
solutionslist=[]
next=[]
for param in paramlist:
  param[1]=param[1].replace(' ', '')
  param[0]=param[0].replace(' ', '')
  param[4]=param[4].replace(' ', '')
  if param[4]=='notready' and param[2]=='none':
    next=param[1].split(",")
  if param[0] in next:
    if param[2]== 'none' and param[4]!= 'notready':
       solutionslist.append(param[4])
    if param[4]=='notready' and param[2]!='none':
       i=0
       paramAtt=0
       paramcritic=0
       parambad=0
       paramSucc=0
       for p in param[2]:
         if(p.find("Att"))!=-1:
           print(param[0])
           paramAtt+=param[3][i]
         if(p.find("Critical"))!=-1:
           print(param[0])
           paramcritic+=param[3][i]
         if(p.find("Bad"))!=-1:
           print(param[0])
           parambad+=param[3][i]
         if(p.find("Succ"))!=-1:
           print(param[0])
           paramSucc+=param[3][i]
         i+=1
         
       if paramAtt-paramSucc>paramSucc and paramAtt!=0 and paramSucc!=0:
         next= param[1].split(",")
       if parambad<paramcritic and parambad!=0 and paramcritic!=0:
         if(param[1].find(","))!=-1:
          next= param[1]
         else:
          next= param[1].split(",")
    #    else:
    #      print('none2')
# print(next)
print(solutionslist)
# new_row1 = {'category': 'MCPC', 'next': 'MCPCparams', 'params': 'none', 'param_value': 0, 'solution': 'not ready'}
# df = pd.DataFrame([new_row1])
# Xdf.loc[len(Xdf)] = new_row1
# new_row1 = {'category': 'MCPCparams', 'next': 'MCPCSolutions', 'params': ['pmCriticalBorderEvalReport','pmBadCovSearchEvalReport'], 'param_value': [], 'solution': 'not ready'}
# df.loc[len(df)] = new_row1
# Xdf.loc[len(Xdf)] = new_row1
# new_row1 = {'category': 'MCPCSolutions', 'next': 'none', 'params': 'none', 'param_value': 0, 'solution': 'Tune parameters according to number of carriers in LTE'}
# Xdf.loc[len(Xdf)] = new_row1
# df.loc[len(df)] = new_row1
# print(df)
# # convert the DataFrame to a dictionary
# data = df.to_dict(orient='records')
# # insert the dictionary into the MongoDB collection
# collection.insert_many(data)
# Remove the 'B' column