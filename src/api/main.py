from fastapi import FastAPI
from api.routes import router

# Initialize the FastAPI application
app = FastAPI(title="LLaMA API", version="1.0.0")

# Include the router to add routes to the application
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """
    Root endpoint to welcome users or redirect to the documentation.
    """
    return {"message": "Welcome to the LLaMA API! Visit /docs for the API documentation."}


if __name__ == "__main__":
    import uvicorn

    # Start the API server
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
