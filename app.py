from flask import Flask
from flask_cors import CORS
import data_hotel


app = Flask(__name__)

CORS(app) 

@app.route("/")
def start_service():
    return "<p>service started! v1</p>"

@app.route('/api/list-hotel', methods=['GET'])
def list_hotel(): 
        res = data_hotel.get_list_hotel()
        return res
