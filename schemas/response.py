from pydantic import BaseModel

class ResponseSchema(BaseModel):
    code: int = 0
    message: str = "success"