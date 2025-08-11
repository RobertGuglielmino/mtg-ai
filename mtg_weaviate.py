
    
import json
import weaviate
from weaviate.classes.init import Auth
import os
from weaviate.classes.config import Configure, Property, DataType
import getpass

    
if not os.environ.get("WEAVIATE_URL"):
  os.environ["WEAVIATE_URL"] = getpass.getpass("WEAVIATE_URL: ")
  
  
if not os.environ.get("WEAVIATE_API"):
  os.environ["WEAVIATE_API"] = getpass.getpass("WEAVIATE_API: ")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=os.environ["WEAVIATE_API"],
)



client.collections.create(
    "MTGOfficialRules3",
    properties=[
        Property(name="rule", data_type=DataType.TEXT),
        Property(name="rule_number", data_type=DataType.TEXT),
        Property(name="section", data_type=DataType.TEXT),
    ],
    vector_config=[
        Configure.Vectors.text2vec_weaviate(
            name="rule_vector",
            source_properties=["rule"],
            model="text2vec-huggingface"
        )
    ]
)
client.collections.create(
    "MTGRulings3",
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="rulings", data_type=DataType.TEXT),
    ],
    vector_config=[
        Configure.Vectors.text2vec_weaviate(
            name="rulings_vector",
            source_properties=["name", "rulings"],
            model="text2vec-huggingface"
        )
    ]
)

client.collections.create(
    "MTGCards3",
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="manaCost", data_type=DataType.TEXT),
        Property(name="type", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT),
        Property(name="power", data_type=DataType.TEXT),
        Property(name="toughness", data_type=DataType.TEXT),
        Property(name="colors", data_type=DataType.TEXT_ARRAY),
    ],
    vector_config=[
        Configure.Vectors.text2vec_weaviate(
            name="cards_vector",
            source_properties=[
                "name",
                "manaCost",
                "colors",
                "type",
                "text",
                "power",
                "toughness",
            ],
            model="text2vec-huggingface"
        )
    ]
)

################################################

official_rules_collection = client.collections.get("MTGOfficialRules3")
rulings_collection = client.collections.get("MTGRulings3")
cards_collection = client.collections.get("MTGCards3")



################################################

rulebook_file = "./mtg_rules.txt"
with open(rulebook_file, 'r', encoding='utf-8') as f:
    rulebook_text = f.read()
    
splitText = rulebook_text.split("\n\n")

with official_rules_collection.batch.fixed_size(batch_size=500) as batch:
    for src_obj in splitText:
        batch.add_object(
            properties={
                "rule": src_obj,
            },
        )
        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break
        
        
################################################
        
 
cards_file = "./AtomicCards.json"
with open(cards_file, 'r', encoding='utf-8') as f:
    cards_text = json.loads(f.read())
    
# print(cards_text)

with rulings_collection.batch.fixed_size(batch_size=500) as batch:
    for key, value in cards_text['data'].items():
        card_info = value[0]
        rulings = []
        if "rulings" in card_info:
            # print(card_info['rulings'])
            # print(card_info['rulings'].keys())
            rulings = [ruling['text'] for ruling in card_info['rulings']]
            # for ruling in card_info["rulings"]:
                # print(rulings)
                
            rulingsStr = ""
            
            for ruling in rulings:
                rulingsStr += str(ruling).replace("\"", "").replace("\'", "")
            print(rulingsStr)
        
            
            batch.add_object(
                properties={
                    "name": card_info['name'],
                    "rulings": rulingsStr,
                },
            )
        
        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break
 
 
 
 
################################################

with cards_collection.batch.fixed_size(batch_size=500) as batch:
    for key, value in cards_text['data'].items():
        card_info = value[0]
        
        batch.add_object(
            properties={
                "name": card_info['name'],
                "manaCost": card_info.get("manaCost", "").replace("{", "").replace("}", ""),
                "type": card_info.get("type", ""),
                "text": card_info.get("text", ""),
                "power": card_info.get("power", ""),
                "toughness": card_info.get("toughness", ""),
                "colors": card_info.get("colors", ""),
            },
        )
        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break

print(client.is_ready())  # Should print: `True`

# Work with Weaviate

client.close()