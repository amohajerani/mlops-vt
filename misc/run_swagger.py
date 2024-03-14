"""
This script runs a local Flask server that serves a Swagger UI for a given API.
The Swagger UI allows you to interact with the API directly from your browser.

The script takes two command-line arguments:
--url: The URL of the API.
--auth_code: The authorization code for the API.

Example usage:
python3 run_swagger.py --url http://4.149.66.177/api/v1/endpoint/vt-online-vtpoc9dev --auth_code HlgJpiRPHT0D72hPBXUN0aaykLVdegA3
"""

import argparse
from flask import Flask, Response, request
from flask_swagger_ui import get_swaggerui_blueprint
import requests
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Swagger UI for a given API.')
parser.add_argument('--url', required=True, help='The URL of the API.')
parser.add_argument('--auth_code', required=True, help='The authorization code for the API.')
args = parser.parse_args()

app = Flask(__name__)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/swagger.json'  # Our Swagger schema

@app.route(API_URL)
def swagger_json():
    """
    Fetch the Swagger JSON from the API and modify it to use HTTP.
    """
    response = requests.get(args.url + '/swagger.json')
    swagger = json.loads(response.content)

    # Modify the scheme to 'http'
    swagger['schemes'] = ['http']

    # Only include the POST /api/v1/endpoint/score endpoint
    swagger['paths'] = {
        '/score': swagger['paths']['/api/v1/endpoint/vt-online-vtpoc9dev/score']
    }

    return Response(json.dumps(swagger), mimetype='application/json')

@app.route('/score', methods=['POST'])
def score():
    """
    Proxy for the API's score endpoint. Adds the authorization code to the headers.
    """
    headers = dict(request.headers)
    headers['Authorization'] = 'Bearer ' + args.auth_code
    response = requests.post(args.url + '/score', headers=headers, json=request.json)
    return Response(response.content, status=response.status_code, mimetype=response.headers['Content-Type'])

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be served at this URL
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(debug=True)