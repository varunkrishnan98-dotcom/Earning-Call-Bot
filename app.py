import streamlit as st
import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional
import PyPDF2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Earnings Call Extractor", layout="wide")
st.title("📊 Earnings Call & Investor Presentation Extractor")
st.markdown("Upload a transcript or presentation (PDF) to automatically extract financial models, catalysts, tone, and macro contexts.")

# --- SECURE CLIENT INITIALIZATION ---
# Fetches the API key from Streamlit Cloud Secrets
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    
    # Initialize the Gemini model with a system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are an expert equity research analyst. Extract the requested metrics strictly from the text provided. If a metric is not mentioned, leave it blank. Do not hallucinate."
    )
    
    # Patch the model with Instructor
    client = instructor.from_gemini(client=model)
    
except KeyError:
    st.error("API Key not found. Please set 'GEMINI_API_KEY' in your Streamlit Secrets.")
    st.stop()

# --- PYDANTIC SCHEMAS ---
class FinancialModelInputs(BaseModel):
    forward_guidance: Optional[str] = Field(description="Revised revenue expectations, EBITDA margins, and EPS targets.")
    revenue_drivers: Optional[str] = Field(description="Volume vs pricing power, pipeline visibility, new contract wins.")
    cost_structure: Optional[str] = Field(description="OpEx, raw material costs, labor inflation, supply chain constraints.")
    capital_allocation: Optional[str] = Field(description="Buybacks, dividend hikes, debt refinancing, CapEx.")

class QualitativeSignals(BaseModel):
    confidence_levels: str = Field(description="Aggressive growth outlooks vs defensive/evasive tone.")
    omissions_and_deflections: List[str] = Field(description="Questions avoided, ignored topics, or delayed timelines.")

class StrategicCatalysts(BaseModel):
    product_market_expansion: List[str] = Field(description="New product rollouts, geographic expansion, pricing strategy.")
    ma_and_restructuring: Optional[str] = Field(description="Potential acquisitions, divestitures, or cost-cutting.")

class MacroContext(BaseModel):
    competitive_dynamics: str = Field(description="Market share claims relative to competitors.")
    macro_headwinds: List[str] = Field(description="Geopolitical challenges, regulatory changes, or consumer softening.")

class QuarterlyEarningsExtraction(BaseModel):
    financial_inputs: FinancialModelInputs
    qualitative_signals: QualitativeSignals
    strategic_catalysts: StrategicCatalysts
    macro_context: MacroContext

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        if page.extract_text():
            extracted_text += page.extract_text() + "\n"
    return extracted_text

def analyze_transcript(text_content):
    """Passes text to the LLM and enforces structured JSON output via Instructor."""
    return client.chat.completions.create(
        response_model=QuarterlyEarningsExtraction,
        messages=[
            {"role": "user", "content": f"Extract data from this transcript:\n\n{text_content}"}
        ],
        max_retries=3
    )

# --- UI WORKFLOW ---
uploaded_file = st.file_uploader("Upload Transcript (PDF)", type=["pdf"])

if uploaded_file is not None:
    if st.button("Extract Insights"):
        with st.spinner("Parsing PDF and extracting financial insights (this may take a minute)..."):
            
            # 1. Parse text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            # Note: For production with 15k+ word docs, chunking logic goes here.
            # We are passing the first 30,000 characters here to prevent token limits on standard tiers.
            text_to_process = raw_text[:30000] 
            
            try:
                # 2. Extract Data
                insights = analyze_transcript(text_to_process)
                
                # 3. Display Results
                st.success("Extraction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Direct Inputs for Financial Models")
                    st.write("**Forward Guidance:**", insights.financial_inputs.forward_guidance)
                    st.write("**Revenue Drivers:**", insights.financial_inputs.revenue_drivers)
                    st.write("**Cost Structure:**", insights.financial_inputs.cost_structure)
                    st.write("**Capital Allocation:**", insights.financial_inputs.capital_allocation)
                    
                    st.subheader("🧩 Strategic Catalysts")
                    st.write("**Product/Market Expansion:**", insights.strategic_catalysts.product_market_expansion)
                    st.write("**M&A / Restructuring:**", insights.strategic_catalysts.ma_and_restructuring)

                with col2:
                    st.subheader("🎙️ Qualitative Signals & Tone")
                    st.write("**Confidence Levels:**", insights.qualitative_signals.confidence_levels)
                    st.write("**Omissions & Deflections:**", insights.qualitative_signals.omissions_and_deflections)
                    
                    st.subheader("🌍 Industry & Macro Context")
                    st.write("**Competitive Dynamics:**", insights.macro_context.competitive_dynamics)
                    st.write("**Macro Headwinds:**", insights.macro_context.macro_headwinds)
                    
            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")
