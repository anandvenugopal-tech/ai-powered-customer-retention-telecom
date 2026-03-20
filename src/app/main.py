from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_churn
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


class CustomerData(BaseModel):
    gender: str
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    senior_citizen: int
    tenure: int
    monthly_charges: float
    total_charges: float

app = FastAPI(
    title = 'AI-Powered Customer Retention System-Telecom',
    version = '1.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return {'message': "API is running"}

@app.post('/predict')
def predict(customer: CustomerData):
    try:
        result = predict_churn(customer.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
def get_connection():
    return psycopg2.connect(
        host = os.getenv('POSTGRES_HOST'),
        user = os.getenv('POSTGRES_USER'),
        database = os.getenv('POSTGRES_DB'),
        password = os.getenv('POSTGRES_PASSWORD')
    )

@app.get('/customer/{customer_id}')
def get_customer(customer_id: str):
    conn = get_connection()
    cursor = conn.cursor()

    query = f"SELECT * FROM fact_customer_profile WHERE customer_id = '{customer_id}';"
    cursor.execute(query)

    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    if row:
        return dict(zip(columns, row))
    return {'error': 'Customer not found'}
    
