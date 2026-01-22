from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "change-this"
    
    # Kaggle
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    
    # Paths
    UPLOAD_DIR: str = "./data/uploads"
    DATASET_DIR: str = "./data/dataset"
    MODEL_DIR: str = "./trained_models"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"

settings = Settings()