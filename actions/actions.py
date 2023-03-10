import datetime as dt 
from typing import Any, Text, Dict, List
import pymongo
import json
import pandas as pd
import pymysql
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


def autocorrect(word,coll,search,sim_threshold=0.6):
    max_sim=0.0
    most_sim_word = word
    if search==1:
        fields = ['abbrv', 'fullname']
    if search==2:
        fields=['BSC','Bande de fréquences','Gouvernorat','Site','Site_Code',"Type d'Installation",'Longitude','Latitude','LAC','Identifiant']
        sim_threshold=0.7
    dbs = set()
    cursor = coll.find({})
    for document in cursor:
        for field in fields:
            if field in document:
                words = set(document[field].split())
                dbs.update(words)
    
    for dw in dbs:
        cur_sim=get_simularity(word,dw)
        if cur_sim>max_sim:
            max_sim= cur_sim
            most_sim_word= dw

    return most_sim_word if max_sim> sim_threshold else "none"


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
                    if autocorrect(x.upper(),collection)!= None:
                        if autocorrect(x,collection)!="none":
                            slot_value=autocorrect(x,collection,search)                       
                if slot_value=="none":
                    dispatcher.utter_message("Can you please write the term that you're looking for again")
                    return {}
                else:
                    query = {
                        "$or": [
                            {"abbrv": {"$regex": slot_value.upper()}},
                            {"fullname": {"$regex": slot_value.upper()}}
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

# class ValidateReseauPhyForm(FormValidationAction):
#     def name(self) -> Text:
#         return "validate_reseau_phy_form"

#     def validate_site(
#         self,
#         slot_value: Any,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: DomainDict,
#     ) -> Dict[Text, Any]:
#         """Validate `site` value."""
#         if slot_value!= None:
#                 # Get the collection
#             collection = db["reseau-physique"]
#             # Find all documents in the collection
#             slot_value=slot_value.replace(" ", "")
#             print(slot_value)
#             aff="Sorry, reseau is yet to be added"
#             documents = collection.find({"$or": [{"Site": slot_value.upper()},
#                                         {"Site_Code": slot_value.upper()},{"Identifiant": slot_value.upper()},{"BSC": slot_value.upper()},{"Bande de fréquences": slot_value.upper()},{"Gouvernorat": slot_value.upper()}
#                                         ,{"HBA(m)": slot_value.upper()},{"LAC": slot_value.upper()},{"Latitude": slot_value.upper()}
#                                         ,{"Longitude": slot_value.upper()},{"Puissance isotrope rayonnée équivalente (PIRE) dans chaque secteur": slot_value.upper()},{"Secteur": slot_value.upper()}
#                                         ,{"Type d'Installation": slot_value.upper()},{"azimut du rayonnement maximum dans chaque secteur": slot_value.upper()}]})
#             print(documents)
#             if documents!=None :
#                 i = 0
#                 aff=""
#                 for document in documents: 
#                     i+=1
#                     aff+=str(i)+" - "+document['Site']+" : "+ document['Identifiant']+" : "+ document['Site_Code']+" : "+ document['LAC']+" : "+ document['Bande de fréquences']+"\n"
#             print(aff)
#             dispatcher.utter_message(aff)
#             dispatcher.utter_message("the number of sites is "+str(i))
#             return {"site": slot_value}
#         else:
#             dispatcher.utter_message(text=f"Can you please write the term that you're looking for again")
#             return {"site": None}

class ActionSolveProblem(Action):

    def name(self) -> Text:
        return "action_problem_solve"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_problem = next(tracker.get_latest_entity_values("problem"), None)
        
        if not current_problem:
            msg = "no problem have been checked"
            dispatcher.utter_message(text=msg)
            return []
        
        
        msg = f"Sure thing! I'll remember that the problem is {current_problem}."
        dispatcher.utter_message(text=msg)
        
        return [SlotSet("problem", current_problem)]
    
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