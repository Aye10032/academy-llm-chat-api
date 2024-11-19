from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta, UTC

app = FastAPI()

SECRET_KEY = "your-secret-key"

class LoginData(BaseModel):
    email: str
    password: str

@app.post("/api/login")
async def login(data: LoginData):
    if data.email == "test@example.com" and data.password == "password":
        token = jwt.encode(
            {
                "sub": data.email,
                "exp": datetime.now(UTC) + timedelta(days=1)
            },
            SECRET_KEY,
            algorithm="HS256"
        )
        return {"token": token}
    else:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")