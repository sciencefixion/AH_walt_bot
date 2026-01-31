from fastapi import FastAPI, HTTPException

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.routers import journals, passages, vector_ops, langgraph_ops,

app = FastAPI()

# Setting CORS (Cross Origin Resource Sharing) policy
origins = ["http://localhost"]

#Global custom exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exception: HTTPException):
    return JSONResponse(
        status_code=exception.status_code,
        content={"message":exception.detail}
    )

# Import routers
app.include_router(journals.router)
app.include_router(passages.router)
app.include_router(vector_ops.router)
app.include_router(langgraph_ops.router)

@app.get("/")
async def read_root():
    return {"message":"Welcome to AutoHagiography with walt_bot!"}