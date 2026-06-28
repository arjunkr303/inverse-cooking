from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.recipe import router as recipe_router

app = FastAPI(
    title="Inverse Cooking API",
    description="Industrial culinary intelligence and recipe generation API.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten this up for your production web servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recipe_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}
