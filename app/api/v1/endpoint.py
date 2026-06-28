from fastapi import APIRouter, HTTPException, status
from app.schemas.recipe import RecipeRequest, RecipeResponse
from app.services.vision import generate_recipe_from_image

router = APIRouter(prefix="/recipes", tags=["Recipes"])

@router.post("/predict", response_model=RecipeResponse, status_code=status.HTTP_200_OK)
async def predict_recipe(payload: RecipeRequest):
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Base64 image payload missing")
        
    recipe_markdown = await generate_recipe_from_image(payload.image_base64)
    return RecipeResponse(recipe=recipe_markdown)
