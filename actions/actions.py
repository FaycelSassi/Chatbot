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


#the show time action
class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_show_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text=f"{dt.datetime.now()}")

        return []

 
 #sending definition
class ActionDefinion(Action):
   
    def name(self) -> Text:
        return "action_ask_definition"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message.get('text')
        # Get the collection
        collection = db["chap1"]
        # Find all documents in the collection
        
        print(message)
        if message != None:
            document = collection.find_one({"$or": [{"abbrv": message.upper()},
                                       {"fullname": message.upper()}]})
        print(document)
        aff=""
        if document==None :
            aff="Sorry, definition is yet to be added"
        else:
            df = pd.DataFrame(document, index=[1])
            aff=document['fullname']+"("+document['abbrv']+") "+document['definition']
            print(aff)
        dispatcher.utter_message(aff)
    
        return []


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
    #  action
    class ActionResetAllSlots(Action):
        def name(self):
            return "action_reset_all_slots"

        def run(self, dispatcher, tracker, domain):
            return [AllSlotsReset()]