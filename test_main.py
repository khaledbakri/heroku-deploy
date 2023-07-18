from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_say_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome!"}

def test_inference_greater_than_50k_case():
    response = client.post("/registers/", 
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
                                    "native_country": "United-States"})
    assert response.status_code == 200
    assert response.json() == {"salary": 1}

def test_inference_less_than_or_equal_to_50k_case():
    response = client.post("/registers/", 
                            json = {"age": 50,
                                    "workclass": "Private",
                                    "fnlgt": 167886,
                                    "education": "Some-college",
                                    "education_num": 10,
                                    "marital_status": "Married-civ-spouse",
                                    "occupation": "Sales",
                                    "relationship": "Husband",
                                    "race": "White",
                                    "sex": "Male",
                                    "capital_gain": 0,
                                    "capital_loss": 0,
                                    "hours_per_week": 40,
                                    "native_country": "United-States"})
    assert response.status_code == 200
    assert response.json() == {"salary": 0}