import datetime as dt 
from typing import Any, Text, Dict, List
import pymysql
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import AllSlotsReset


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
        cursor.close()
        connection.close()
        dispatcher.utter_message(aff)

                
        return []


#validate existence
def clean_name(name):
    connection = pymysql.connect(
            host='localhost',
            user='root',
            password='password',
            db='rasabot',
        )
    cursor = connection.cursor()
    sql = "SELECT * FROM `defs` WHERE `abbrv`= '"+ name.upper()+"' OR `fullname`='"+name.upper()+"';"
    cursor.execute(sql)
    result = cursor.fetchone()
    if(result == None) : 
        return "".join([c for c in name if c.isalpha()])

#the add definition action
class ActionAddDefinition(Action):
    def name(self) -> Text:
            return "action_add_def"
            
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        
        print("in")
        fullname = tracker.get_slot("fullname")
        full_def= tracker.get_slot("full_def")
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='password',
            db='rasabot',
        )
        x = fullname.split("#")
        abbrv=""
        if len(x)==1:
            abbrv=x[0]
        else:
            for i in (x):
                abbrv+=i[0]
        cursor = connection.cursor()
        sql = "INSERT INTO `defs`(`abbrv`, `fullname`, `definition`) VALUES('"+abbrv.upper()+"','"+fullname.upper()+"','"+full_def+"')"
        print(sql)
        cursor.execute(sql)
        connection.commit()
        cursor.close()
        connection.close()
        dispatcher.utter_message("definition added")
        return []
    #the reset all slots action
    #  action
    class ActionResetAllSlots(Action):
        def name(self):
            return "action_reset_all_slots"

        def run(self, dispatcher, tracker, domain):
            return [AllSlotsReset()]