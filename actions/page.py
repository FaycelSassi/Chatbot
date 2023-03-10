from flask import Flask, render_template, request
import pymongo
import json
#setting up the connection to the database
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Get the database
db = client["rasa"]
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    BSC = request.form['BSC']
    fréquences = request.form['fréquences']
    Gouvernorat = request.form['Gouvernorat']
    HBA = request.form['HBA']
    Identifiant = request.form['Identifiant']
    LAC = request.form['LAC']
    Latitude = request.form['Latitude']
    Longitude = request.form['Longitude']
    antennes = request.form['antennes']
    cellules = request.form['cellules']
    porteuses = request.form['porteuses']
    PIRE = request.form['PIRE']
    Secteur = request.form['Secteur']
    Site = request.form['Site']
    Site_Code = request.form['Site_Code']
    Tits = request.form['Tits']
    Type = request.form['Type']
    azimut = request.form['azimut']

    return f"Hello {BSC}, your frequency is {fréquences} and your Gouvernorat is: {Gouvernorat}"
@app.route('/feature')
def featureindex():
    return render_template('featureform.html')

@app.route('/submitfeature', methods=['POST'])
def submitfeature():
    collection = db["chap1"]
    abbrv = request.form['abbrv']
    feature = request.form['feature']
    Definition = request.form['Definition']
    Others = request.form['Others']
    document = collection.find_one({"$or": [{"abbrv": abbrv.upper()},
                                       {"fullname": feature.upper()}]})
    if(document != None) :
        result="Hello, this feature ,"+document['abbrv']+", already exists, your feature is "+document['fullname'] +"and your definition is: "+document['definition']+" for more info :"+document['others']
    else:
        result="Will be added soon"
        return render_template("result.html",result = result)

if __name__ == '__main__':
    app.run(debug=True, port=4949)