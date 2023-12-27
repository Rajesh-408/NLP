import uuid
import hashlib
import datetime
from gpt4all import GPT4All

from src.settings import db, config, logger
model = GPT4All(model_name='orca-mini-3b-gguf2-q4_0.gguf')

class GPTDescription():
    def __init__(self, benefit_name, srvc_def_desc):
        self.benefit_name = benefit_name
        self.srvc_def_desc = srvc_def_desc

    def get_description_id(self):
        hashed_benefit_name = hashlib.md5(self.benefit_name.encode()).hexdigest()
        description_id = str(uuid.UUID(hashed_benefit_name))
        return description_id

    def generate_gpt_description(self):
        if self.benefit_name == self.srvc_def_desc:
            medical_term = self.benefit_name
        else:
            medical_term = f"{self.srvc_def_desc}, {self.benefit_name}"
        with model.chat_session():
            response1 = model.generate(prompt='hello, assume yourself as a virutal chatbot supporting customer of a healthcare company.', temp=0)
            response2 = model.generate(prompt=f'Can you generate description for following MEDICAL benefit: {medical_term} , I want only in 1 simple sentence', temp=0)
            #response3 = model.generate(prompt='thank you', temp=0)
            #print(model.current_chat_session)
        #return {"version_1":response2,"version_2": response2.split(":")[1]}
        return response2.split(":")[1]


def generate_new_description(benefit_name, srvc_def_desc):
    collection = db[config.get('mongo_collection_name')]
    gpt_obj = GPTDescription(benefit_name=benefit_name, srvc_def_desc= srvc_def_desc)
    description_id = gpt_obj.get_description_id()
    data = {"description_id": description_id,
            "benefit_name": benefit_name,
            "timestamp": datetime.datetime.utcnow(),
            "new_description": True,
            "srvc_def_desc": srvc_def_desc
            }
    description = gpt_obj.generate_gpt_description()
    descriptions_dict = {}
    market_segements = config["market_segements"]
    for market_segement in market_segements:
        descriptions_dict[market_segement] = {"description": description,
                                              "description_status": "in review"}
    data['markets'] = descriptions_dict
    collection.insert_one(data)

    return description_id
