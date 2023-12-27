import logging
import sys
from flask import request, make_response, Flask, jsonify
# from anthem.tenx.utils.config import GLOBAL_CONFIG
# from anthem.tenx.utils.log import SplunkLogger
import json
import os

from src.settings import config, logger

from src.utilities.generate_new_descriptions import GPTDescription
from src.decorators.environment_decorators import disable_for_environment


# logging.setLoggerClass(SplunkLogger)

app = Flask(__name__)
# logger = logging.getLogger("app")

@app.route("/downloadGPT4ALLModel", methods=['GET'])
@disable_for_environment('downloadGPT4ALLModel')
def download_gpt4all_model():
    try: 
        req = request.get_json()
        benefit_name = req.get("benefitName")
        srvc_def_desc = req.get("srvcDefDesc")
        obj = GPTDescription(benefit_name, srvc_def_desc)
        return jsonify(obj.generate_gpt_description())
    except Exception as e:
        traceback_str = logging.traceback.format_exc()
        logger.error(traceback_str)
        return str(e), 404

@app.route("/hello")
def hello():
    app.logger.info("Hello World found")
    logger.info("message", thomas='cool')
    return "Hello World"

@app.route("/configprops")
def configprops():
    logger.info("SPLUNK-TRACE", request="GET", func="myfunc")
    print(config)
    return config


@app.route("/refresh", methods=['POST'])
def refresh():
    return config.load()


@app.route('/ping', methods=['GET'])
def ping():
    """
    Check if API Alive
    ---
    tags:
      - Check Alive
    consumes:
      - application/json
    produces:
      - application/json
      - text/xml
      - text/html
    responses:
        200:
            description: Success
            schema:
            id: return_test
            properties:
        500:
            description: Error
    """
    release_tag_name = os.environ.get('release_tag_name')
    return jsonify(status='Alive', releaseTag=release_tag_name),200


 
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
if __name__ == "__main__":
    app.run() 
