from flask_cors import CORS
from flask import Flask
import json
import torch
from flask import Blueprint,request, jsonify

from LSTM_Inference.main_run import lstm_run

app=Flask(__name__)
CORS(app)

from .utils import logging

bp=Blueprint('lstm_api',__name__)

@bp.route("/api/run-lstm", methods=["POST"]) #Web Link Endpoint
def scrape():
    try:
        data=lstm_run() #Runs python file from backend
        print("DEBUG: lstm_run returned:", data)
        return jsonify({"status":"success","data":data})
    except Exception as e:
        logging.error(f"Failed to run lstm {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True, port=5000)