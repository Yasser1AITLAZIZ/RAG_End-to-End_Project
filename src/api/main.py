from fastapi import FastAPI
from api.routes import router
from utils.logger import setup_logger
import logging

# Initialize logger
api_logger = setup_logger(name="api_logger", log_file="logs/api.log", level=logging.INFO)

# Initialize the FastAPI application
app = FastAPI(title="LLaMA API", version="1.0.0")

# Include the router to add routes to the application
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    api_logger.info("API server is starting...")

@app.on_event("shutdown")
async def shutdown_event():
    api_logger.info("API server is shutting down...")

@app.get("/")
async def root():
    """
    Root endpoint to welcome users or redirect to the documentation.
    """
    return {"message": "Welcome to the LLaMA API! Visit /docs for the API documentation."}


if __name__ == "__main__":
    import uvicorn
    
    # Log server start
    api_logger.info("Starting API server...")
    # Start the API server
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
