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
class ValidateSimplePizzaForm(FormValidationAction):
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
        # Get the collection
        collection = db["chap1"]
        # Find all documents in the collection
        slot_value=slot_value.replace(" ", "")
        print(slot_value)
        aff="Sorry, definition is yet to be added"
        if slot_value != None:
            document = collection.find_one({"$or": [{"abbrv": slot_value.upper()},
                                       {"fullname": slot_value.upper()}]})
            print(document)
            if document!=None :
                aff=document['fullname']+"("+ document['abbrv']+") "+ document['definition']
                if document['others']!="":
                    aff=document['fullname']+"("+ document['abbrv']+") "+ document['definition']+" for more information: "+document['others']
        else:
            dispatcher.utter_message(text=f"Can you please write the term that you're looking for again")
            return {"definition": None}
        print(aff)
        dispatcher.utter_message(aff)
        return {"definition": slot_value}


#validate existence
def clean_name(name):
    collection = db["chap1"]
    document = collection.find_one({"$or": [{"abbrv": name.upper()},
                                       {"fullname": name.upper()}]})
    if(document == None) : 
        return "".join([c for c in name if c.isalpha()])

#the add definition action
class ActionAddDefinition(Action):
    def name(self) -> Text:
            return "action_add_def"
            
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the collection
        collection = db["chap1"]
        print("in")
        fullname = tracker.get_slot("fullname")
        full_def= tracker.get_slot("full_def")
        x = fullname.split("#")
        abbrv=""
        if len(x)==1:
            abbrv=x[0]
        else:
            for i in (x):
                abbrv+=i[0]
        new_document = {
            "abbrv": abbrv.upper(),
            "fullname": fullname.upper(),
            "definition": full_def,
            "others": ""
        }
        result = collection.insert_one(new_document)
        if(result.inserted_id!= None):
            dispatcher.utter_message("definition added")
        else:
            dispatcher.utter_message("there was a problem with the addition please try again")
        return []

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
            document = collection.find_one({"$or": [{"Site": slot_value.upper()},
                                        {"Site_Code": slot_value.upper()},{"Identifiant": slot_value.upper()}]})
            if document!=None :
                aff=document['Site']+" : "+ document['Identifiant']+" : "+ document['Site_Code']+" : "+ document['LAC']+" : "+ document['Bande de fréquences']
                print(aff)
            dispatcher.utter_message(aff)
            return {"site": slot_value}
        else:
            dispatcher.utter_message(text=f"Can you please write the term that you're looking for again")
            return {"site": None}