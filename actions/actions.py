import datetime as dt 
from typing import Any, Text, Dict, List
import pymysql
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
        return "action_ask_definition"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #setting up the connection to the database

        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='password',
            db='rasabot',
        )
        message = tracker.latest_message.get('text')

        #creating a cursor to execute SQL commands
        cursor = connection.cursor()
        # Query data   
        print(message)
        if message == None:
            sql="empty"
        else: sql = "SELECT * FROM `defs` WHERE `abbrv`= '"+ message.upper()+"' OR `fullname`='"+message.upper()+"';"
        cursor.execute(sql)
        result = cursor.fetchone()
        aff=""
        if result==None :
            aff="Sorry, definition is yet to be added"
        else: aff=result[2]+"("+result[1]+") "+result[3]
        
        dispatcher.utter_message(aff)

                
        return []
        