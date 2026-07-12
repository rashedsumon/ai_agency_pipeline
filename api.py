import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Internal orchestration imports
from src.data_loader import AgencyDataLoader
from src.model import VideoProductionAgent
from src.utils import convert_production_sheet_to_dataframe

# Initialize FastAPI App instance
app = FastAPI(
    title="AI Agency Automation Hub API",
    description="Production-grade endpoints to orchestrate automated script generation and multi-modal prompting structures via LangGraph.",
    version="1.0.0"
)

# Initialize reusable processing components
data_loader = AgencyDataLoader()
agent_instance = VideoProductionAgent()

# --- Pydantic Schema Specifications ---
class IngestRequest(BaseModel):
    raw_context: str = Field(
        default_factory=data_loader.load_raw_real_life_example,
        description="Raw context text or summaries extracted from source documents/NotebookLM."
    )

class SlideRow(BaseModel):
    Slide: int
    On_Screen_Text: str = Field(..., alias="On-Screen Text")
    Visual_Asset_Prompt: str = Field(..., alias="Visual Asset Prompt")
    Voice_Over_Audio_Text: str = Field(..., alias="Voice-Over Audio Text")

    class Config:
        populate_by_name = True

class PipelineResponse(BaseModel):
    status: str
    video_script: str
    production_sheet: List[SlideRow]

# --- Core API Endpoint Logic ---
@app.post("/api/v1/generate-pipeline", response_model=PipelineResponse)
async def generate_production_pipeline(payload: IngestRequest):
    """
    Ingests high-density raw topic sources, moves them through a deterministic 
    LangGraph agent sequence, and surfaces structured asset scripts for automation engines.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API Key configuration missing. Please declare the OPENAI_API_KEY environment variable."
        )

    try:
        # Construct target state configuration matrix
        initial_state = {
            "raw_context": payload.raw_context,
            "video_script": "",
            "production_sheet": []
        }
        
        # Compile and invoke active Graph network execution
        graph = agent_instance.compile_pipeline_graph()
        final_output_state = graph.invoke(initial_state)
        
        # Extract operational assets from pipeline state
        raw_sheet_data = final_output_state.get("production_sheet", [])
        
        return PipelineResponse(
            status="success",
            video_script=final_output_state.get("video_script", ""),
            production_sheet=raw_sheet_data
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed during engine traversal execution: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Confirms operational and architectural readiness of systemic endpoints."""
    return {"status": "healthy", "engine": "LangGraph Active"}

# Entry runner invocation layout
if __name__ == "__main__":
    import uvicorn
    # In production, run via CLI: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)