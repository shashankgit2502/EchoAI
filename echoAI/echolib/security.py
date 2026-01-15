
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import settings
from .types import UserContext

security = HTTPBearer(auto_error=False)

class AuthError(HTTPException):
    def __init__(self, detail: str = 'Unauthorized'):
        super().__init__(status_code=401, detail=detail)

def create_token(sub: str, email: str, *, expires_minutes: int = 60) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        'iss': settings.jwt_issuer,
        'aud': settings.jwt_audience,
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=expires_minutes)).timestamp()),
        'sub': sub,
        'email': email,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm='HS256')

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=['HS256'], audience=settings.jwt_audience, issuer=settings.jwt_issuer)
    except Exception:
        return None

async def user_context(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> UserContext:
    if creds is None:
        raise AuthError()
    payload = decode_token(creds.credentials)
    if not payload:
        raise AuthError()
    return UserContext(user_id=payload['sub'], email=payload.get('email',''))
