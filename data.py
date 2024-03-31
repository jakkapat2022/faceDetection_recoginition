import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
#---------------------------------------------------------------------------------------------------------#
cred = credentials.Certificate("system-of-face-recognition-firebase-adminsdk-b959m-a1287026ef.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': 'https://system-of-face-recognition-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
opendata = db.reference('2022-06-25/ingkaew')
opendata.set({
    'date':'2022-06-25',
    'name':'Singkaew singkaew',
    'time':'000'
})