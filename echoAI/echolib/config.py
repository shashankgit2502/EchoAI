
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    app_name: str = os.getenv('APP_NAME', 'echo-mermaid-platform')
    service_mode: str = os.getenv('SERVICE_MODE', 'mono')  # mono | micro
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    jwt_secret: str = os.getenv('JWT_SECRET', 'dev-secret-change-me')
    jwt_issuer: str = os.getenv('JWT_ISSUER', 'echo')
    jwt_audience: str = os.getenv('JWT_AUDIENCE', 'echo-clients')

settings = Settings()
