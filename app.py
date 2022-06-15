from flask import Flask, request, jsonify
from flask_cors import CORS
import data_hotel
import model_rekomendasi


app = Flask(__name__)

CORS(app) 

# Route API Home
@app.route("/")
def start_service():
    return "<p>service started! v1</p>"

# Route API List Hotel
@app.route('/api/list-hotel', methods=['GET'])
def list_hotel(): 
        res = data_hotel.get_list_hotel()
        return res

# Route API POST Data
@app.route('/api/rekomendasi-hotel', methods=['POST'])
def rekomendasi_hotel(): 
        # res = file_model.function_model(request json yg diinput)
        res = model_rekomendasi.rekomendasi_hotel(data=request.json)
        
        # return data dalam bentuk json
        return res
