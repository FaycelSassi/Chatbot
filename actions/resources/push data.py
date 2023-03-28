import pandas
doc = pandas.read_excel("../resources/LteFeatures1.xlsx")
print(doc.head())
doc.rename(columns={"Feature Name": "fullname","Feature Identity": "abbrv","Comments":"others"}, inplace=True)
doc['definition'] ="is in package name :"+ doc['Value Package Name'] + " ( "+doc['Value Package Identity']+" ) with License Control MO/DU radio Node,OptionalFeatureLicense= " +doc['License Control MO\nDU radio Node,\nOptionalFeatureLicense=\n']+ " and License Control MO/Baseband-based Node, FeatureState= "+doc['License Control MO\nBaseband-based Node,\nFeatureState=\n'] +" and featureState/Recommended Value =  "+doc['featureState\nRecommended Value']
# remove columns
doc = doc.drop(columns=['Value Package Name', 'Value Package Identity','License Control MO\nDU radio Node,\nOptionalFeatureLicense=\n','License Control MO\nBaseband-based Node,\nFeatureState=\n','featureState\nRecommended Value'])
doc = doc.reindex(["abbrv","fullname","definition","others"], axis=1)
# drop all rows with null values
doc = doc.dropna()
print(doc.isna().sum())
print( doc.shape[0])
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["rasa"]
collection = db["chap1"]
data = doc.to_dict(orient='records')
collection.insert_many(data)

