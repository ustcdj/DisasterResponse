from flask import Flask

app = Flask(__name__)

# app is folder, run is in app folder
from app import run
