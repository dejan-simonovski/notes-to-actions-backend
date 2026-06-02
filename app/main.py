from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import meeting_router
from app.exceptions.meeting_exception import MeetingException, meeting_exception_handler

# Instantiate FastAPI application
app = FastAPI(
    title="AI Meeting Notes to Action Items API",
    description=(
        "A robust, 100% async, stateless AI utility service backend "
        "designed to summarize meeting notes and extract Eisenhower Matrix-prioritized "
        "action items with Slack integrations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register global custom exceptions handler
app.add_exception_handler(MeetingException, meeting_exception_handler)

# Include meeting routes router
app.include_router(meeting_router.router)

@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """
    Standard service health check endpoint.
    """
    return {"status": "healthy", "service": "meeting-notes-api"}

if __name__ == "__main__":
    import uvicorn
    # Start the local development server when file is run directly
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
