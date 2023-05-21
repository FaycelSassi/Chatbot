import datetime as dt
from typing import Any, Text, Dict, List
import pymongo
import re
from io import BytesIO
import base64
import numpy as np
import difflib
import numpy as np
from scipy import stats
from pymongo import MongoClient, UpdateOne
import pandas as pd
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
#from keras.callbacks import EarlyStopping
from keras.layers import ConvLSTM2D
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.events import SlotSet

# setting up the connection to the database
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Get the database
db = client["rasa"]


def create_bigram(w):
    return [w[i]+w[i+1] for i in range(len(w)-1)]


def get_simularity(w1, w2):
    w1, w2 = w1.lower(), w2.lower()
    common = []
    bigram1, bigram2 = create_bigram(w1), create_bigram(w2)
    for i in range(len(bigram1)):
        try:
            cmn_elt = bigram2.index(bigram1[i])
            common.append(bigram1[i])
        except:
            continue
    return len(common)/max(len(bigram1), len(bigram2), 1)


from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer


def autocorrect(input_word, coll,search,k=1):
     # This function takes a word and a list of valid words, and returns the closest match to the input word from the list of valid words.
    threshold = 0.6
    if search == 1:
        fields = ['abbrv', 'fullname']
    if search == 2:
        fields = ['BSC', 'Bande de fréquences', 'Gouvernorat', 'Site', 'Site_Code',
                  "Type d'Installation", 'Longitude', 'Latitude', 'LAC', 'Identifiant']
    cursor = coll.find({})
    distances = []
    dbs = set()
    if search!= 3:
        for document in cursor:
            for field in fields:
                if field in document:
                    word_list = set(document[field].split())
                    dbs.update(word_list)
        # Get the list of similar words with their similarity score
        similar_words = [(w, get_simularity(input_word, w)) for w in dbs]

        # Find the word with the highest similarity score above the threshold
        best_word = max(
            similar_words, key=lambda x: x[1] if x[1] >= threshold else -1)
        print(best_word)
        if best_word[1] < threshold:
            return "none"
        return best_word[0]
    if search == 3:
        fields = ['ERBS Id']
    dbs=[]
    for document in cursor:
        for field in fields:
            if field in document:
                word_list = str(document[field]).upper()
                dbs.append(word_list)
    return difflib.get_close_matches(input_word, dbs, n=k)

# def autocorrect(word, coll, search):

#     # This function takes a word and a list of valid words, and returns the closest match to the input word from the list of valid words.
#     threshold = 0.6
#     if search == 1:
#         fields = ['abbrv', 'fullname']
#     if search == 2:
#         fields = ['BSC', 'Bande de fréquences', 'Gouvernorat', 'Site', 'Site_Code',
#                   "Type d'Installation", 'Longitude', 'Latitude', 'LAC', 'Identifiant']
#     if search==3:
#         fields = ['ERBS Id']
#     distances = []
#     dbs = set()
#     cursor = coll.find({})
#     for document in cursor:
#         for field in fields:
#             if field in document:
#                 word_list = set(document[field].split())
#                 dbs.update(word_list)
#      # Get the list of similar words with their similarity score
#     similar_words = [(w, get_simularity(word, w)) for w in dbs]

#     # Find the word with the highest similarity score above the threshold
#     best_word = max(
#         similar_words, key=lambda x: x[1] if x[1] >= threshold else -1)
#     print(best_word)
#     if best_word[1] < threshold:
#         return "none"
#     return best_word[0]

 # sending definition


class ValidateDefForm(Action):
    def name(self) -> Text:
        return "Action_Validate_feature"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> list[Dict[Text, Any]]:
        slot_values = tracker.latest_message.get("text")
        search = 1
        aff = "Sorry, feature is yet to be added"
        if slot_values == None:
            dispatcher.utter_message("No terms were added")
            return {}
        else:
            # Get the collection

            collection = db["chap1"]
            slot_value = "none"
            for x in slot_values.split():
                if autocorrect(x.upper(), collection, search) != None:
                    if autocorrect(x, collection, search) != "none":
                        slot_value = autocorrect(x, collection, search)
            if slot_value == "none":
                dispatcher.utter_message(
                    "Can you please write the term that you're looking for again")
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
                        aff = str(i) + " - " + document['abbrv'] + " : " + \
                            document['fullname'] + " " + document['definition']
                        if document['others'] != "":
                            aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + \
                                " " + \
                                document['definition'] + \
                                " for more information: " + document['others']
                        print(aff)
                        dispatcher.utter_message(aff)
                        i += 1
                dispatcher.utter_message(
                    "number of features found with "+slot_value+" is "+str(i-1))
                return {}


# validate existence
def clean_name(name):
    collection = db["chap1"]
    document = collection.find_one({"$or": [{"abbrv": name.upper()},
                                            {"fullname": name.upper()}]})
    if (document == None):
        return "".join([c for c in name if c.isalpha()])


# the reset all slots action
class ActionResetAllSlots(Action):
    def name(self):
        return "action_reset_all_slots"

    def run(self, dispatcher, tracker, domain):
        return [AllSlotsReset()]

 # sending definition


class ActionSiteInfo(Action):
    def name(self) -> Text:
        return "action_site_info"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> list[Dict[Text, Any]]:
        slot_values = tracker.latest_message.get("text")
        print(slot_values)
        search = 2
        aff = "Sorry, physical is yet to be added"
        if slot_values == None:
            dispatcher.utter_message("No terms were added")
            return {}
        else:
            # Get the collection
            collection = db["reseau-physique"]
            slot_value = "none"
            for x in slot_values.split():
                print(x)
                print(autocorrect(x.upper(), collection, search))
                if autocorrect(x.upper(), collection, search) != None:
                    if autocorrect(x.upper(), collection, search) != "none":
                        slot_value = autocorrect(x.upper(), collection, search)
            if slot_value == "none":
                dispatcher.utter_message("Physical network can't be found")
                return {}
            else:
                documents = collection.find({"$or": [{"Site": slot_value.upper()},
                                                     {"Site_Code": slot_value.upper()}, {"Identifiant": slot_value.upper()}, {"BSC": slot_value.upper()}, {"Bande de fréquences": slot_value.upper()}, {"Gouvernorat": slot_value.upper()}, {"HBA(m)": slot_value.upper()}, {"LAC": slot_value.upper()}, {"Latitude": slot_value.upper()}, {"Longitude": slot_value.upper()}, {"Puissance isotrope rayonnée équivalente (PIRE) dans chaque secteur": slot_value.upper()}, {"Secteur": slot_value.upper()}, {"Type d'Installation": slot_value.upper()}, {"azimut du rayonnement maximum dans chaque secteur": slot_value.upper()}]})
                print(documents)
                if documents != None:
                    i = 0
                    aff = ""
                    for document in documents:
                        i += 1
                        aff += str(i)+" - "+document['Site']+" : " + document['Identifiant']+" : " + \
                            document['Site_Code']+" : " + document['LAC'] + \
                            " : " + document['Bande de fréquences']+"\n"
                    print(aff)
                    dispatcher.utter_message(aff)
                    dispatcher.utter_message(
                        "the number of sites with "+slot_value+" is "+str(i))
                    return {}


def findsolution(dispatcher,x):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["rasa"]
    collection = db["Chap2"]
    data = list(collection.find())
    # create a Pandas DataFrame from the list of dictionaries
    Xdf = pd.DataFrame(data)
    if Xdf.empty:
        print("please add data to the database")
        return {}
    Xdf = Xdf.drop('_id', axis=1)
    # print(Xdf)
    paramlist = Xdf.values.tolist()
    solutionslist = []
    next = []
    collection = db['Compteurs']
        # retrieve documents with specified columns
    documents = collection.find({})
    df = pd.DataFrame(list(documents))
    # print(df)
    k = 0
    j = 0
    # param[0]== category 
    # param[1]== next
    # param[2]== solution
    # param[3]==formule de calcul de résultat des conteurs
    # param[4]==liste des conteurs à utiliser
    for param in paramlist:
        param[1] = param[1].replace(' ', '')
        param[0] = param[0].replace(' ', '')
        # param[2] = param[2].replace(' ', '')
        if param[2] == 'not ready' and param[4] == 'none':
            next = param[1].split(",")
        if param[0] in next:
            if param[4] == 'none' and param[2] != 'not ready':
                solutionslist.append(param[4])
            if param[2] == 'not ready' and param[4] != 'none':
                for x in param[4]:
                    if x.replace(" ", "")  in df.columns:
                        k = 1
                if k == 1:
                    i = param[4]
                    i = [s.replace(' ', '') for s in i]
                    if (i[0]in df.columns) :
                        j += 1
                        result = df.eval(param[3])
                        df["resultat "+param[0]] = result
    s = []
    for j in range(len(df)):
        s.append('')
    solution = []
    sol = ""
    solist = []
    # Determine a solution to the problems
    next = ""
    for i in range(len(df)):
        for col in df.columns:
            if df[col][i] == True:
                if "resultat" in col:
                    solist = []
                    x = col.replace("resultat ", "")
                    my_string = ''
                    for param in paramlist:
                        if param[0] == x:
                            next = param[1]
                        if next != "":
                            for y in next.split(","):
                                if y == param[0]:
                                    if param[2] not in solution:
                                        solution.append(param[2])
                    my_string = ' '.join(solution)
                    my_string = 'for '+x+' : '+my_string
                    if my_string not in solution:
                        solist.append(my_string)
                    sol = ''.join(solist)
                    if sol not in s[i]:
                        s[i] = s[i]+sol+' ;\n '
                    next = ""
    df['solution'] = s
    # select columns A and C by name using loc, and convert to a list
    selected_columns = df.loc[:, ['ERBS Id', 'solution']].values.tolist()
    i=0
    s=[]
    for col in selected_columns:
        if col[1] != '':
            i+=1
            s.append(col[0])
    my_string = ' , '.join(s) 
    dispatcher.utter_message("there are problems in "+str(i)+" sites : "+my_string)
    j=0
    s=df['solution']
    id=df['ERBS Id']
    for i in id:
        collection.update_many(
                {'ERBS Id':i},
                {"$set":
                    {
                        "Solution": s[j]
                    }
                })
        j+=1


# the reset all slots action
class ActionProblemSolve(Action):
    def name(self):
        return "action_problem_solve"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        findsolution(dispatcher,0)
        return {}

# the reset all slots action

class ActionInfoProblem(Action):
    def name(self):
        return "action_Info_Problem"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        collection = db['Compteurs']
        # retrieve documents with specified columns
        documents = collection.find({})
        df = pd.DataFrame(list(documents))
        selected_columns = df.loc[:, ['ERBS Id', 'solution']].values.tolist()
        for col in selected_columns:
            if col[1] != '':
                dispatcher.utter_message(col[0]+':\n'+col[1])
        return{}

class ActionSiteProblem(Action):
    def name(self):
        return "action_Site_Problem"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["rasa"]
        collection = db["Traffic"]
        msg=tracker.latest_message.get("text")
        delimiters = ["for ", "of "]
        regex_pattern = '|'.join(map(re.escape, delimiters))
        split_string = re.split(regex_pattern, msg)[1]
        split_string=split_string.replace(" ","_")
        if '4G' not in split_string:
            split_string='4G_'+split_string
        print(split_string)
        split_string=split_string.upper()
        print(split_string)
        site=autocorrect(split_string,collection,3)
        print(site)
        if len(site)!=0:
            site=site[0]
        collection = db['Compteurs']
        if site!="none":
            # retrieve documents with specified columns
            documents = collection.find({})
            df = pd.DataFrame(list(documents))
            if 'Solution' not in df.columns: 
                findsolution(dispatcher,0)
            selected_columns = df.loc[:, ['ERBS Id', 'Solution']].values.tolist()
            for col in selected_columns:
                if col[0].upper() == site:
                    if col[1]=="":
                        dispatcher.utter_message("there's no problem with "+col[0])
                    else:   
                        dispatcher.utter_message('the solution for '+col[0]+' is : '+ col[1])

                    return [SlotSet("site", col[0])]
        else:  
            dispatcher.utter_message('please verify input')
            return [SlotSet("site", site)] 

class ActionProblemSolveML(Action):
    def name(self):
        return "action_problem_solve_ML"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # retrieve all documents from the collection
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["rasa"]
        collection = db["Chap2"]
        data = list(collection.find())
        # create a Pandas DataFrame from the list of dictionaries
        Xdf = pd.DataFrame(data)
        Xdf = Xdf.drop('_id', axis=1)
        paramlist = Xdf.values.tolist()
        solutionslist = []
        next = []
        collection = db['Conteurs']
        # retrieve documents with specified columns
        documents = collection.find({})
        df = pd.DataFrame(list(documents))
        k = 0
        j = 0
        for param in paramlist:
            param[1] = param[1].replace(' ', '')
            param[0] = param[0].replace(' ', '')
            param[2] = param[2].replace(' ', '')
            if param[2] == 'notready' and param[4] == 'none':
                next = param[1].split(",")
            if param[0] in next:
                if param[4] == 'none' and param[2] != 'notready':
                    solutionslist.append(param[4])
                if param[2] == 'notready' and param[4] != 'none':
                    for x in param[4]:
                        if x.replace(" ", "") not in df.columns:
                            print(x + " values not added in database")
                        else:
                            k = 1
                    if k == 1:
                        i = param[4]
                        if len(i) == 2:
                            j += 1
                            i[0].replace(" ", "")
                            i[1].replace(" ", "")
                            result = df.apply(lambda row: eval(param[3], {i[0].replace(" ", ""): row[i[0].replace(
                                " ", "")], i[1].replace(" ", ""): row[i[1].replace(" ", "")]}), axis=1)
                            df["resultat "+param[0]] = result
            # get the names of columns that contain 'resultat'
        s = []
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
        # Keep only columns with 'result' or 'solution' in their names
        filtered_columns = [col for col in df.columns if 'resultat' in col or 'solution' in col]
        dfknn = df[filtered_columns]                    
        # # Convert boolean columns to numeric (0 or 1)
        for x in df.columns:
            if "resultat" in x:
                dfknn[x] = dfknn[x].astype(int)
        # # Repeat each row 3 times
        dfknn = dfknn.loc[df.index.repeat(4)].reset_index(drop=True)
        solution=dfknn['solution']
        dfknn= dfknn.drop('solution', axis=1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            dfknn, solution, test_size=0.4, random_state=42)

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)
        # Get the current time and calculate the time an hour ago
        now = datetime.now()
        one_hour_ago  = datetime.now() - timedelta(hours=3)
        filter_df = df[df['created_at'] > one_hour_ago]
        # Keep only columns with 'result' in their names
        filtered_columns = [col for col in filter_df.columns if 'resultat' in col]
        filtered_df = filter_df[filtered_columns]
        results=knn.predict(filtered_df)
        filter_df['solution']=results
        print(filter_df)
        # Define a function to convert a row to a string
        def row_to_string(row):
            return ', '.join(row.astype(str))
        # filter by the "id" and "created_at" columns
        if filter_df.empty:
            dispatcher.utter_message("no values added in the last hour")
        strings = filter_df.apply(row_to_string, axis=1)
        # Print the resulting strings
        
        # create a string of all columns in the dataframe
        cols_str = ', '.join(df.columns)

        # print the string
        dispatcher.utter_message(cols_str)
        for index, row_string in strings.items():
             dispatcher.utter_message("Row "+str(index+1)+":"+ row_string)
        # Evaluate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        dispatcher.utter_message('Accuracy: ', str(accuracy))
        return{}
    
class ActionPredictTraffic(Action):
    def name(self):
        return "action_predict_traffic_ML"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["rasa"]
        collection = db["Traffic"]
        msg=tracker.latest_message.get("text")
        delimiters = ["for", "of"]
        regex_pattern = '|'.join(map(re.escape, delimiters))
        split_string = re.split(regex_pattern, msg)[1]
        split_string='4G'+split_string
        split_string=split_string.upper()
        site=autocorrect(split_string,collection,3)
        print(site)
        if len(site)!=0:
            site=site[0]
            print(site)
            dispatcher.utter_message("Predictions for : "+site)

            # Connect to MongoDB
            client = pymongo.MongoClient("mongodb://localhost:27017/")
            db = client["rasa"]

            # Create a new collection
            mycol = db["Traffic"]

            # Retrieve all data from the collection
            data = mycol.find()
            result_traffic= pd.DataFrame(list(data))
            result_traffic['ERBS Id']=result_traffic['ERBS Id'].str.upper()
            site_data = result_traffic[result_traffic['ERBS Id'] == site]
            data_to_sum = site_data[['Date', 'Hour','Trafic PS (Gb)']]
            grouped_data = data_to_sum.groupby(['Date', 'Hour']).sum()
            # group data by date and time_type
            grouped_data=grouped_data.reset_index('Hour')
            grouped_data=grouped_data.drop("Hour",axis=1)

            grouped_data = grouped_data[(np.abs(stats.zscore(grouped_data['Trafic PS (Gb)'])) < 3)]
                    #Convert pandas dataframe to numpy array
            dataset = grouped_data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train_size = int(len(dataset) * 0.66)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            #creates a dataset where X is the number of passengers at a given time (t, t-1, t-2...) 
            #and Y is the number of passengers at the next time (t + 1).

            def to_sequences(dataset, seq_size=1):
                x = []
                y = []

                for i in range(len(dataset)-seq_size-1):
                    #print(i)
                    window = dataset[i:(i+seq_size), 0]
                    x.append(window)
                    y.append(dataset[i+seq_size, 0])
                    
                return np.array(x),np.array(y)
                

            seq_size = 120  # Number of time steps to look back 
            #Larger sequences (look further back) may improve forecasting.

            trainX, trainY = to_sequences(train, seq_size)
            testX, testY = to_sequences(test, seq_size)



            print("Shape of training set: {}".format(trainX.shape))
            print("Shape of test set: {}".format(testX.shape))
            trainX = trainX.reshape((trainX.shape[0], 1, 1, 1, seq_size))
            testX = testX.reshape((testX.shape[0], 1, 1, 1, seq_size))

            model = Sequential()
            model.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='relu', input_shape=(1, 1, 1, seq_size)))
            model.add(Flatten())
            model.add(Dense(1))
            model.add(Dense(32))
            model.add(Dense(1))
            model.add(Dense(32))
            model.add(Dense(1))
            model.add(Flatten())
            model.compile(optimizer='Nadam', loss='mean_squared_error')
            print(model.summary())
            r2=0.5
            i=0
            while(i<4):
                i+=1
                if(r2<0.65):
                    model.fit(trainX, trainY, validation_data=(testX, testY),
                    verbose=2, epochs=50)

                    # make predictions

                    trainPredict = model.predict(trainX)
                    testPredict = model.predict(testX)
                    # invert predictions back to prescaled values
                    #This is to compare with original input values
                    #SInce we used minmaxscaler we can now use scaler.inverse_transform
                    #to invert the transformation.
                    trainPredict = scaler.inverse_transform(trainPredict)
                    trainY = scaler.inverse_transform([trainY])
                    testPredict = scaler.inverse_transform(testPredict)
                    testY = scaler.inverse_transform([testY])

                    # calculate root mean squared error
                    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                    print('Train Score: %.2f RMSE' % (trainScore))
                    dispatcher.utter_message('Train Score par RMSE: '+str(trainScore))
                    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                    print('Test Score: %.2f RMSE' % (testScore))
                    dispatcher.utter_message('Test Score par RMSE: ' + str(testScore))
                    r2 = r2_score(testY[0], testPredict[:,0])
                    print('R2 score:', r2)
                    dispatcher.utter_message('R2 score: '+ str(r2))
                    # shift train predictions for plotting
                    #we must shift the predictions so that they align on the x-axis with the original dataset. 
                    trainPredictPlot = np.empty_like(dataset)
                    trainPredictPlot[:, :] = np.nan
                    trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

                    # shift test predictions for plotting
                    testPredictPlot = np.empty_like(dataset)
                    testPredictPlot[:, :] = np.nan
                    testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict

                    # plot baseline and predictions
                    plt.figure(figsize=(20,10))
                    plt.plot(scaler.inverse_transform(dataset))
                    plt.plot(trainPredictPlot)
                    plt.plot(testPredictPlot)
                    plt.legend(['Data', 'Train Predict result', 'Test Predict result'])
                    # Save the plot to a temporary buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    # Encode the image as base64
                    image = base64.b64encode(buf.read()).decode('utf-8')

                    # Send the image using the dispatcher
                    dispatcher.utter_message(image=f"data:image/png;base64,{image}")
        return [SlotSet("site", site)]