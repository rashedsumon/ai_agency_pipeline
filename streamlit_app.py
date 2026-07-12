import os
import streamlit as st
import pandas as pd

# Structural Import Configuration
from src.data_loader import AgencyDataLoader
from src.model import VideoProductionAgent, execute_stub_hf_fine_tune
from src.utils import convert_production_sheet_to_dataframe

# Page setup Configuration
st.set_page_config(page_title="AI Agency Automation Hub", page_icon="🎬", layout="wide")

st.title("🚀 Web Design to AI Agency Production Pipeline")
st.subheader("Automated PPTX Slide Content to Social Media Video Prompt Sheet Generator")

# Initialize Pipeline Modules
data_loader = AgencyDataLoader()
agent_instance = VideoProductionAgent()

# Create tabs for structured evaluation matching interview targets
tab_production, tab_hf_scaffolding, tab_system_logs = st.tabs([
    "🎥 Live Video Production Engine", 
    "🤗 Hugging Face Dataset & Fine-Tuning Setup", 
    "📋 Technical Reference Guide"
])

with tab_production:
    st.write("### Stage 1 & 2: Ingest Raw Summaries & Orchestrate Prompt Extraction via LangGraph")
    
    # Pre-populate raw text based on the Metformin real-life example
    default_input = data_loader.load_raw_real_life_example()
    user_input_area = st.text_area("Input Data (NotebookLM Consolidated Topic Sources):", value=default_input, height=200)
    
    if st.button("Execute Automated Production Graph"):
        if not os.environ.get("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"):
            st.warning("⚠️ Application running in sandbox/simulation mode. Provide an OPENAI_API_KEY environment variable to enable live connection to model nodes.")
        
        with st.spinner("LangGraph agent executing node sequences..."):
            # Initial state mapping execution
            initial_state = {
                "raw_context": user_input_area,
                "video_script": "",
                "production_sheet": []
            }
            
            # Compile and run Graph state
            graph = agent_instance.compile_pipeline_graph()
            final_output_state = graph.invoke(initial_state)
            
            st.success("✅ Automation Run Complete!")
            
            # Displaying Structured Model Output results cleanly 
            st.write("#### Generated Short-Form Video Script")
            st.info(final_output_state.get("video_script", "No text script returned."))
            
            st.write("#### Stage 3 Model Output: Standardized Bulk CSV Table for Canva / CapCut Automation")
            sheet_data = final_output_state.get("production_sheet", [])
            df_sheet = convert_production_sheet_to_dataframe(sheet_data)
            
            if not df_sheet.empty:
                st.dataframe(df_sheet, use_container_width=True)
                
                # Ready mapping generation to simulate Canva bulk drop
                csv_bytes = df_sheet.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Bulk Creation CSV for Canva/CapCut",
                    data=csv_bytes,
                    file_name="canva_bulk_production_sheet.csv",
                    mime="text/csv"
                )

with tab_hf_scaffolding:
    st.write("### Model Personalization and Scale Management Scaffolding")
    st.markdown(
        "To break dependency on generalized models, the agency can automate collection of "
        "historical successful scripts directly from Hugging Face datasets to fine-tune local models."
    )
    
    hf_dataset_repo = st.text_input("Hugging Face Dataset Repository Target:", value="imdb")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Auto-Download & Cache Dataset"):
            with st.spinner("Fetching stream packets from Hugging Face hub..."):
                df_downloaded = data_loader.auto_download_hf_dataset(hf_dataset_repo)
                st.write("### Extracted Source Frame (First 5 records):")
                st.dataframe(df_downloaded, use_container_width=True)
                st.success("Dataset successfully cached locally for downstream optimization processing.")
                
    with col2:
        if st.button("Trigger Local Model Fine-Tune Routine"):
            target_csv = os.path.join(data_loader.local_dir, "downloaded_pipeline_templates.csv")
            if not os.path.exists(target_csv):
                # Ensure validation dataset structure is present
                df_fallback = pd.DataFrame({"text": ["Example fine-tuning prompt string content mapping setup." for _ in range(5)]})
                df_fallback.to_csv(target_csv, index=False)
                
            with st.spinner("Initializing Hugging Face Transformers training context loop..."):
                status_msg = execute_stub_hf_fine_tune(target_csv)
                st.code(status_msg, language="text")
                st.success("Scaffolding active. Ready for deployment configuration with enterprise cluster GPUs.")

with tab_system_logs:
    st.write("### Agency Reference Architecture Mapping")
    st.markdown("""
    #### Execution Flow Mapping
    1. **NotebookLM Extract Ingestion**: Raw context maps directly to `PipelineState["raw_context"]`.
    2. **LangGraph Consolidation Node**: Executes granular compression layer, keeping layout descriptions under strict word caps.
    3. **LangGraph Multi-Modal Prompt Mapping Node**: Inspects output from previous node state and builds structured prompt sheets matching Midjourney / Veo constraints.
    4. **DataFrame Flat Wrapper Extraction**: System formats state metrics directly into layout tables ready for immediate ingestion into standard video template variables.
    """)