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

#setting up the connection to the database
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Get the database
db = client["rasa"]


 #sending definition
class ValidateDefForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_def_form"

    def validate_definition(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `definition` value."""
        slot_value=slot_value.replace(" ", "")
        print(slot_value)
        aff="Sorry, feature is yet to be added"
        if slot_value == None:
            dispatcher.utter_message("Can you please write the term that you're looking for again")
            return {"definition": None}
        else:
            # Get the collection
            collection = db["chap1"]# perform the query
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
            return {"definition": slot_value}


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

class ValidateReseauPhyForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_reseau_phy_form"

    def validate_site(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `site` value."""
        if slot_value!= None:
                # Get the collection
            collection = db["reseau-physique"]
            # Find all documents in the collection
            slot_value=slot_value.replace(" ", "")
            print(slot_value)
            aff="Sorry, reseau is yet to be added"
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
            return {"site": slot_value}
        else:
            dispatcher.utter_message(text=f"Can you please write the term that you're looking for again")
            return {"site": None}