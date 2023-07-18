import requests

url = 'https://deploy-model-dpvu.onrender.com/registers'
json = {"age": 22,
                                    "workclass": "Private",
                                    "fnlgt": 31387,
                                    "education": "Bachelors",
                                    "education_num": 13,
                                    "marital_status": "Married-civ-spouse",
                                    "occupation": "Adm-clerical",
                                    "relationship": "Own-child",
                                    "race": "Amer-Indian-Eskimo",
                                    "sex": "Female",
                                    "capital_gain": 2885,
                                    "capital_loss": 0,
                                    "hours_per_week": 25,
                                    "native_country": "United-States"}

response = requests.post(url, json=json)
print("Model inference:", response.json())
print("Status code:", response.status_code)