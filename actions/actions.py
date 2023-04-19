import datetime as dt 
from typing import Any, Text, Dict, List
import pymongo
import numpy as np
import json
import pandas as pd
import pymysql
import os

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.events import SlotSet

#setting up the connection to the database
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Get the database
db = client["rasa"]

def create_bigram(w):
    return[w[i]+w[i+1] for i in range(len(w)-1)]

def get_simularity(w1,w2):
    w1,w2=w1.lower(),w2.lower()
    common=[]
    bigram1,bigram2= create_bigram(w1),create_bigram(w2)
    for i in range(len(bigram1)):
        try:
            cmn_elt=bigram2.index(bigram1[i])
            common.append(bigram1[i])
        except:
            continue
    return len(common)/max(len(bigram1),len(bigram2),1)


def autocorrect(word,coll,search):

    # This function takes a word and a list of valid words, and returns the closest match to the input word from the list of valid words.
    threshold =0.6
    if search==1:
        fields = ['abbrv', 'fullname']
    if search==2:
        fields=['BSC','Bande de fréquences','Gouvernorat','Site','Site_Code',"Type d'Installation",'Longitude','Latitude','LAC','Identifiant']
    distances = []
    dbs = set()
    cursor = coll.find({})
    for document in cursor:
        for field in fields:
            if field in document:
                word_list = set(document[field].split())
                dbs.update(word_list)
     # Get the list of similar words with their similarity score
    similar_words = [(w, get_simularity(word, w)) for w in dbs]

    # Find the word with the highest similarity score above the threshold
    best_word = max(similar_words, key=lambda x: x[1] if x[1] >= threshold else -1)
    print(best_word)
    if best_word[1] < threshold: 
        return "none"
    return best_word[0]


 #sending definition
class ValidateDefForm(Action):
    def name(self) -> Text:
        return "Action_Validate_feature"
    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> list[Dict[Text, Any]]:
            slot_values=tracker.latest_message.get("text")
            search=1
            aff="Sorry, feature is yet to be added"
            if slot_values == None:
                dispatcher.utter_message("No terms were added")
                return {}
            else:        
                # Get the collection

                collection = db["chap1"]
                slot_value="none"
                for x in slot_values.split():
                    if autocorrect(x.upper(),collection,search)!= None:
                        if autocorrect(x,collection,search)!="none":
                            slot_value=autocorrect(x,collection,search)                       
                if slot_value=="none":
                    dispatcher.utter_message("Can you please write the term that you're looking for again")
                    return {}
                else:
                    query = {
                        "$or": [
                            {"abbrv": {"$regex": slot_value}},
                            {"fullname": {"$regex": slot_value}}
                        ]
                    }
                    documents = collection.find(query)
                    # loop through the matching documents and print their fields
                    i = 1
                    if collection.count_documents(query) > 0: 
                        for document in documents:
                            aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition']
                            if document['others'] != "":
                                aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition'] + " for more information: " + document['others']
                            print(aff)
                            dispatcher.utter_message(aff)
                            i += 1   
                    dispatcher.utter_message("number of features found with "+slot_value+" is "+str(i-1))
                    return {}
        

#validate existence
def clean_name(name):
    collection = db["chap1"]
    document = collection.find_one({"$or": [{"abbrv": name.upper()},
                                       {"fullname": name.upper()}]})
    if(document == None) : 
        return "".join([c for c in name if c.isalpha()])
         



#the reset all slots action
class ActionResetAllSlots(Action):
    def name(self):
            return "action_reset_all_slots"

    def run(self, dispatcher, tracker, domain):
            return [AllSlotsReset()]

    
 #sending definition
class ActionSiteInfo(Action):
    def name(self) -> Text:
        return "action_site_info"
    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> list[Dict[Text, Any]]:
            slot_values=tracker.latest_message.get("text")
            print(slot_values)
            search=2
            aff="Sorry, physical is yet to be added"
            if slot_values == None:
                dispatcher.utter_message("No terms were added")
                return {}
            else:        
                # Get the collection
                collection = db["reseau-physique"]
                slot_value="none"
                for x in slot_values.split():
                    print(x)
                    print(autocorrect(x.upper(),collection,search))
                    if autocorrect(x.upper(),collection,search)!= None:
                        if autocorrect(x.upper(),collection,search)!="none":
                            slot_value=autocorrect(x.upper(),collection,search)                      
                if slot_value=="none":
                    dispatcher.utter_message("Physical network can't be found")
                    return {}
                else:
                    documents = collection.find({"$or": [{"Site": slot_value.upper()},
                                        {"Site_Code": slot_value.upper()},{"Identifiant": slot_value.upper()},{"BSC": slot_value.upper()},{"Bande de fréquences": slot_value.upper()},{"Gouvernorat": slot_value.upper()}
                                        ,{"HBA(m)": slot_value.upper()},{"LAC": slot_value.upper()},{"Latitude": slot_value.upper()}
                                        ,{"Longitude": slot_value.upper()},{"Puissance isotrope rayonnée équivalente (PIRE) dans chaque secteur": slot_value.upper()},{"Secteur": slot_value.upper()}
                                        ,{"Type d'Installation": slot_value.upper()},{"azimut du rayonnement maximum dans chaque secteur": slot_value.upper()}]})
                    print(documents)
                    if documents!=None :
                        i = 0
                        aff=""
                        for document in documents: 
                            i+=1
                            aff+=str(i)+" - "+document['Site']+" : "+ document['Identifiant']+" : "+ document['Site_Code']+" : "+ document['LAC']+" : "+ document['Bande de fréquences']+"\n"
                        print(aff)
                        dispatcher.utter_message(aff)
                        dispatcher.utter_message("the number of sites with "+slot_value+" is "+str(i))
                        return {}
                    
#the reset all slots action
class ActionProblemSolve(Action):
    def name(self):
            return "action_problem_solve"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
            # file_path = os.path.join(os.path.dirname(__file__), 'LteParams.xlsx')
            # params = pd.read_excel(file_path)
            # retrieve all documents from the collection
            client = pymongo.MongoClient("mongodb://localhost:27017/")
            db = client["rasa"]
            collection = db["chap2"]
            data = list(collection.find())
            # create a Pandas DataFrame from the list of dictionaries
            Xdf = pd.DataFrame(data)
            if Xdf == None:
                dispatcher.utter("please add data to the database")
                return {}
            Xdf = Xdf.drop('_id', axis=1)
            # print(Xdf)
            paramlist = Xdf.values.tolist()
            solutionslist=[]
            next=[]
            collection = db['Conteurs']
            # retrieve documents with specified columns
            documents = collection.find({})
            df = pd.DataFrame(list(documents))
            k=0
            j=0
            for param in paramlist:
                param[1]=param[1].replace(' ', '')
                param[0]=param[0].replace(' ', '')
                param[2]=param[2].replace(' ', '')
                if param[2]=='notready' and param[4]=='none':
                    next=param[1].split(",")
                if param[0] in next:
                    if param[4]== 'none' and param[2]!= 'notready':
                        solutionslist.append(param[4])
                    if param[2]=='notready' and param[4]!='none':
                        for x in param[4]:
                            if x.replace(" ", "") not in df.columns:
                                dispatcher.utter_message(x + " values not added in database")
                            else: k = 1
                        if k==1:
                            i=param[4]
                            if len(i)== 2:
                                j+=1
                                i[0].replace(" ", "") 
                                i[1].replace(" ", "")
                                result = df.apply(lambda row: eval(param[3], {i[0].replace(" ", ""): row[i[0].replace(" ", "")], i[1].replace(" ", ""): row[i[1].replace(" ", "")]}), axis=1) 
                                df["resultat "+param[0]]=result
            s=[]
            for j in range(len(df)):
                s.append('')                   
            solution=[]
            sol=""
            solist=[]
            #Determine a solution to the problems
            next=""   
            for i in range(len(df)):
                for col in df.columns:
                    if df[col][i]==False:
                        if "resultat" in col:
                            solist=[] 
                            x=col.replace("resultat ","")
                            my_string=''
                            for param in paramlist:
                                if param[0]==x:
                                    next=param[1]
                                if next!="" :
                                    for y in next.split(","):
                                        if y==param[0]:
                                            if param[2] not in solution:
                                                solution.append(param[2])              
                            my_string = ' '.join(solution)
                            my_string='for '+x+' : '+my_string 
                            if my_string not in solution:
                                solist.append(my_string)
                            sol=''.join(solist)
                            if sol not in s[i]:
                                s[i]=s[i]+sol+' ;\n '
                            next=""
            df['solution']=s
            # select columns A and C by name using loc, and convert to a list
            selected_columns = df.loc[:, ['OSS_ID', 'SN','ELEMENT','MOID','solution']].values.tolist()
            for col in selected_columns:
                if col[4]!='':
                    dispatcher.utter_message(col[0]+" - "+col[1]+" : \n"+col[4]+" - ")
            
            return {}      

#the reset all slots action
class ActionProblemSolveML(Action):
    def name(self):
            return "action_problem_solve_ML"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            import pandas as pd
            import pymongo
            # retrieve all documents from the collection
            client = pymongo.MongoClient("mongodb://localhost:27017/")
            db = client["rasa"]
            collection = db["Chap2"]
            data = list(collection.find())
            # create a Pandas DataFrame from the list of dictionaries
            Xdf = pd.DataFrame(data)
            Xdf = Xdf.drop('_id', axis=1)
            paramlist = Xdf.values.tolist()
            solutionslist=[]
            next=[]
            collection = db['Conteurs']
            # retrieve documents with specified columns
            documents = collection.find({})
            df = pd.DataFrame(list(documents))
            k=0
            j=0
            for param in paramlist:
                param[1]=param[1].replace(' ', '')
                param[0]=param[0].replace(' ', '')
                param[2]=param[2].replace(' ', '')
                if param[2]=='notready' and param[4]=='none':
                    next=param[1].split(",")
                if param[0] in next:
                    if param[4]== 'none' and param[2]!= 'notready':
                        solutionslist.append(param[4])
                    if param[2]=='notready' and param[4]!='none':
                        for x in param[4]:
                            if x.replace(" ", "") not in df.columns:
                             print(x + " values not added in database")
                            else: k = 1
                        if k==1:
                            i=param[4]
                            if len(i)== 2:
                                j+=1
                                i[0].replace(" ", "") 
                                i[1].replace(" ", "")
                                result = df.apply(lambda row: eval(param[3], {i[0].replace(" ", ""): row[i[0].replace(" ", "")], i[1].replace(" ", ""): row[i[1].replace(" ", "")]}), axis=1) 
                                df["resultat "+param[0]]=result
                # get the names of columns that contain 'resultat'
            s=[]
            for j in range(len(df)):
                s.append('')
            solution=[]
            sol=""
            solist=[]
            #Determine a solution to the problems
            next=""   
            for i in range(len(df)):
                for col in df.columns:
                    if df[col][i]==False:
                        if "resultat" in col:
                            solist=[] 
                            x=col.replace("resultat ","")
                            my_string=''
                            for param in paramlist:
                                if param[0]==x:
                                    next=param[1]
                                if next!="" :
                                    for y in next.split(","):
                                        if y==param[0]:
                                            if param[2] not in solution:
                                                solution.append(param[2])              
                            my_string = ' '.join(solution)
                            my_string='for '+x+' : '+my_string 
                            if my_string not in solution:
                                solist.append(my_string)
                            sol=''.join(solist)
                            if sol not in s[i]:
                                s[i]=s[i]+sol+' ;\n '
                            next=""
            dfknn=df.iloc[:, 9:12]
            # Convert boolean columns to numeric (0 or 1)
            dfknn['resultat HandoverExecutionFailure'] = dfknn['resultat HandoverExecutionFailure'].astype(int)
            dfknn['resultat MCPCparams'] = dfknn['resultat MCPCparams'].astype(int)
            # Repeat each row 3 times
            dfknn = dfknn.loc[df.index.repeat(4)].reset_index(drop=True)
            solution=dfknn['solution']
            dfknn= dfknn.drop('solution', axis=1)
            # We used pmHoExeSucc, pmHoExeAtt, pmCriticalBorderEvalReport, and pmBadCovSearchEvalReport as features. using iloc
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(dfknn,solution, test_size=0.4, random_state=42)

            # Scale the data using StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train a KNN classifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = knn.predict(X_test)

            # Evaluate the accuracy of the classifier
            accuracy = accuracy_score(y_test, y_pred)
            print('Accuracy:', accuracy)
         

