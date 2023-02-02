import datetime as dt 
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


#the show time action
class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_show_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text=f"{dt.datetime.now()}")

        return []

 
 #sending definitionf
class ActionDefinion(Action):
   
    def name(self) -> Text:
        return "action_give_def"
  
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
 
        # play rock paper scissors
        user_choice = tracker.get_slot("definition")
        dispatcher.utter_message(text=f"You chose {user_choice}")

 
        if user_choice == "tcp" :
            dispatcher.utter_message(text="Transmission Control Protocol (TCP)  is one of the main protocols of the Internet protocol suite. It originated in the initial network implementation in which it complemented the Internet Protocol. Therefore, the entire suite is commonly referred to as TCP/IP.")
        if user_choice == "nli" :
            dispatcher.utter_message(text="Nli definition")
        
        return []