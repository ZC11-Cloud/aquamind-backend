from fastapi import FastAPI
from routers.user import router as user_router
app = FastAPI()
app.include_router(user_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
