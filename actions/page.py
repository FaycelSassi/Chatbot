from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('HTML/form.html')

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

    return f"Hello {BSC}, your email is {fréquences} and your message is: {Gouvernorat}"

if __name__ == '__main__':
    app.run(debug=True, port=4949)