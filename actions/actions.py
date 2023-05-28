import datetime as dt
from typing import Any, Text, Dict, List
import pymongo
import re
from sklearn.cluster import KMeans
from io import BytesIO
import base64
import numpy as np
import difflib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
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
        fields = ['Gouvernorat', 'Site', 'Site_Code',
                  "Type d'Installation", 'Longitude', 'Latitude', 'LAC', 'Identifiant']
    cursor = coll.find({})
    distances = []
    dbs = set()
    if search!= 3:
        for document in cursor:
            for field in fields:
                if field in document:
                    word_list = set(str(document[field]).split())
                    dbs.update(word_list)
        # Get the list of similar words with their similarity score
        similar_words = [(w, get_simularity(input_word, w)) for w in dbs]

        # Find the word with the highest similarity score above the threshold
        best_word = max(similar_words, key=lambda x: x[1] if x[1] >= threshold else -1)
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
            print(slot_value)
            if slot_value == "none":
                dispatcher.utter_message(
                    "Please verify the value sent")
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
                    if documents.count()<=10:
                        for document in documents:
                            aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition']
                            if document['others'] != "":
                                aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition'] +" for more information: " + document['others']
                            print(aff)
                            dispatcher.utter_message(aff)
                            i += 1
                        dispatcher.utter_message(
                    "number of features found with "+slot_value+" is "+str(i-1))
                    else: 
                        names=[]
                        i = 1
                        folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "save folder")
                        os.makedirs(folder_path, exist_ok=True)
                        folder_path=folder_path+'/features_'+slot_value+'.txt'
                        file= open(folder_path, 'w')
                        for document in documents:
                            
                            # Save the DataFrame as a CSV file in the new folder.
                            aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition']+"\n"
                            if document['others'] != "":
                                aff = str(i) + " - " + document['abbrv'] + " : " + document['fullname'] + " " + document['definition'] +" for more information: " + document['others']+"\n"
                            
                            file.write(aff)
                            names.append(document['abbrv'])
                            i += 1
                        file.close() 
                        dispatcher.utter_message("The file is saved in "+ folder_path)
                        my_string = ' - '.join(names)
                        dispatcher.utter_message(my_string)
                        dispatcher.utter_message(
                    "number of features found with "+slot_value+" is "+str(i-1))
                return {}



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
                documents = collection.find({"$or": [{"Site": slot_value},
                                                     {"Site_Code": slot_value}, {"Identifiant": slot_value}, {"BSC": slot_value}, {"Bande de fréquences": slot_value}, {"Gouvernorat": slot_value}, {"HBA(m)": slot_value}, {"LAC": slot_value}, {"Latitude": slot_value}, {"Longitude": slot_value}, {"Puissance isotrope rayonnée équivalente (PIRE) dans chaque secteur": slot_value}, {"Secteur": slot_value}, {"Type d'Installation": slot_value}, {"azimut du rayonnement maximum dans chaque secteur": slot_value}]})
                print(documents)
                if documents != None:
                    if documents.count()<=10:
                        i = 0
                        aff = ""
                        for document in documents:
                            i += 1
                            # Remove the _id field from the document.
                            document = {**document, "_id": None}
                            aff = str(document)
                            dispatcher.utter_message(aff)
                        dispatcher.utter_message(
                            "the number of sites with "+slot_value+" is "+str(i))
                    else:
                        aff=[]
                        liste=[]
                        for document in documents:
                            aff.append(document['Site_Code'])
                            # Remove the _id field from the document.
                            document = {**document, "_id": None}
                            liste.append(document)
                        my_string = ' - '.join(aff)
                        dispatcher.utter_message(my_string)
                        df = pd.DataFrame(liste)
                        print(df)
                        folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "save folder")
                        os.makedirs(folder_path, exist_ok=True)
                        # Save the DataFrame as a CSV file in the new folder.
                        df.to_csv(os.path.join(folder_path, slot_value+'_result.csv'), index=False)
                        dispatcher.utter_message("The file is saved in "+ folder_path)
                        dispatcher.utter_message(
                            "the number of sites with "+slot_value+" is "+str(documents.count()))       
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
    new_column_names = df.columns.str.strip("(%)")


    df.columns = new_column_names
    new_column_names = df.columns.str.replace(" ",'')
    df.columns = new_column_names
    new_column_names
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
                    if (i[0] in df.columns) :
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
            if df[col][i] == False:
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
    selected_columns = df.loc[:, ['ERBSId', 'solution']].values.tolist()
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
    id=df['ERBSId']
    folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "save folder")
    os.makedirs(folder_path, exist_ok=True)
        # Save the DataFrame as a CSV file in the new folder.
    df=df.drop("_id",axis=1)
    df.to_csv(os.path.join(folder_path, 'KPI_Validation.csv'), index=False)
    dispatcher.utter_message("The validation file is saved in "+folder_path)
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
        collection = db['Compteurs']
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
        
sia = SentimentIntensityAnalyzer()
def sentiment_analysis(sentence):
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] > 0.3:
                return 'good'
            elif sentiment['compound'] < 0.01:
                return 'low'
            else:
                return 'normal'

class ActionClassifySiteML(Action):
    def name(self):
        return "action_classify_site_ML"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        sia = SentimentIntensityAnalyzer()
        sentence=tracker.latest_message.get("text")
        resultsent = sentiment_analysis(sentence)
        print(resultsent)

        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["rasa"]
        collection = db["Traffic"]
        data = list(collection.find())
        # create a Pandas DataFrame from the list of dictionaries
        result_traffic = pd.DataFrame(data)
        # Create a new collection
        mycol = db["KPIs"]


        # Retrieve all data from the collection
        data = mycol.find()
        result= pd.DataFrame(list(data))
        result=result.fillna(0)
        result_traffic=result_traffic.drop(['Hour','_id','EUtranCell Id','Date'],axis=1)
        
        result=result.fillna(0)
        result['Accessibility'] = result[['S1 Sig Succ Rate', 'RRC Setup Succ Rate', 'E-RAB Estab Succ Rate']].mean()
        print(result['Accessibility'])
        result = result.drop(['_id','S1 Sig Succ Rate', 'RRC Setup Succ Rate','EUtranCell Id', 'E-RAB Estab Succ Rate'], axis=1)
        grouped = result.groupby('ERBS Id').mean()
        grouped_traffic = result_traffic.groupby('ERBS Id').sum()  
        df = pd.concat([grouped, grouped_traffic], axis=1)
        df=df.sort_values(['Trafic PS (Gb)'])
        df=df.reindex()
        X = df[['Trafic PS (Gb)']].values
        # Create a KMeans model with 3 clusters
        model = KMeans(n_clusters=3)

        # Fit the model to the data
        model.fit(X)

        # Get the cluster labels
        labels = model.labels_
        df['cluster'] = labels
        # Calculate mean for each cluster
        cluster_stats = df.groupby('cluster')['Trafic PS (Gb)'].mean()

        # Create a dictionary to map old cluster labels to new ones
        label_map = dict(zip(cluster_stats.sort_values().index, range(3)))

        # Use the dictionary to map the old labels to the new ones
        df['cluster'] = df['cluster'].map(label_map)
        # Map the cluster labels to performance levels
        df['profitability'] = df['cluster'].map({0: 'low', 1: 'normal',2: 'good'})
        selected_df = df[df['profitability'] == resultsent]
        selected_df=selected_df[(selected_df['Call Drop Rate']<0.5) | (selected_df['Accessibility'] <98)]
        # Create a new folder on the desktop
        folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "save folder")
        os.makedirs(folder_path, exist_ok=True)
        # Save the DataFrame as a CSV file in the new folder.
        selected_df.to_csv(os.path.join(folder_path, 'Degraded KPIs + '+resultsent+' Traffic.csv'), index=False)
        # Filter the dataframe to include only rows where performance equals good
        selected_df =selected_df.reset_index('ERBS Id')
        name_string = selected_df['ERBS Id'].to_string(index=False)

        print(name_string)
        dispatcher.utter_message("the sites with a "+resultsent+" profitability and a degraded KPI are : ")
        dispatcher.utter_message(name_string)
        dispatcher.utter_message("the csv file containing the results is in "+ folder_path)

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
                

            seq_size = 24  # Number of time steps to look back 
            #Larger sequences (look further back) may improve forecasting.

            

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
                    trainX, trainY = to_sequences(train, seq_size)
                    testX, testY = to_sequences(test, seq_size)
                    print("Shape of training set: {}".format(trainX.shape))
                    print("Shape of test set: {}".format(testX.shape))
                    trainX = trainX.reshape((trainX.shape[0], 1, 1, 1, seq_size))
                    testX = testX.reshape((testX.shape[0], 1, 1, 1, seq_size))
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
                    
                    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                    print('Test Score: %.2f RMSE' % (testScore))
                    
                    r2 = r2_score(testY[0], testPredict[:,0])
                    print('R2 score:', r2)
                    
                    # shift train predictions for plotting
                    #we must shift the predictions so that they align on the x-axis with the original dataset. 
                    
            if(r2>0.65):
                dispatcher.utter_message('Train Score par RMSE: '+str(trainScore))
                dispatcher.utter_message('Test Score par RMSE: ' + str(testScore))
                dataset=scaler.inverse_transform(dataset)
                dispatcher.utter_message('R2 score: '+ str(r2))
                #forecast
                prediction = [] #Empty list to populate later with predictions
                dates = [] #Empty list to populate later with dates
                current_batch = test[-seq_size:,0] #Final data points in test 
                last_datetime = grouped_data.index[-1]
                current_batch = current_batch.reshape(1, 1, 1, 1, seq_size) #Reshape
                ## Predict future, beyond test dates
                future = 24 #Times
                for i in range( future):
                    current_pred = model.predict(current_batch)[0]
                    prediction.append(current_pred)
                    new_datetime = last_datetime + pd.DateOffset(hours=i+1)
                    dates.append(new_datetime)
                    new_value = np.array([[[[current_pred]]]])
                    # remove the first value
                    current_batch = current_batch[:, :, :, :, 1:]
                    # add the new value at the end
                    current_batch = np.concatenate((current_batch, new_value), axis=4)
                prediction= scaler.inverse_transform(prediction) #inverse to get the actual values
                s1 = pd.DataFrame(dataset, index=grouped_data.index)
                df=pd.DataFrame(prediction,index=dates)
                plt.figure(figsize=(17,10))
                plt.plot(s1)
                plt.plot(df)
                plt.title("Traffic Prediction")
                plt.xlabel("Datetime")
                plt.ylabel("Traffic Volume")
                plt.legend(('Actual', 'Predicted'))
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                # Encode the image as base64
                image = base64.b64encode(buf.read()).decode('utf-8')
                folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "save folder")
                os.makedirs(folder_path, exist_ok=True)
                # Save the DataFrame as a CSV file in the new folder.
                df=df.reset_index()
                df.to_csv(os.path.join(folder_path, site+'Prediction_result.csv'), index=False)
                dispatcher.utter_message("the site predictions are saved in : "+ folder_path)
                # Send the image using the dispatcher
                dispatcher.utter_message(image=f"data:image/png;base64,{image}")
                return [SlotSet("site", site)]
            else:
                dispatcher.utter_message('R2 score: '+ str(r2) + ' is too low to make a prediction please choose a different site')    
                return [SlotSet("site", site)]