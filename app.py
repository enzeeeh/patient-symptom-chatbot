import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
import glob
import datetime

# If using Gemini API via Google Generative AI SDK
def call_gemini_api(messages: List[Dict[str, str]], api_key: str) -> Dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(messages)
    return response.text

# Prompt template for Gemini
PROMPT_TEMPLATE = '''
You are a medical triage assistant. Given the following patient message, do the following:
1. Extract all symptoms mentioned by the patient.
2. Analyze the symptoms and provide likelihood percentages for possible conditions (must sum to 100%).
3. Classify the risk level as: low, moderate, or high.
4. Provide specific recommendations.
5. Reference appropriate medical guidelines.

Respond in this exact JSON format:
{{
    "symptoms": ["symptom1", "symptom2", "symptom3"],
    "conditions": [
        {{"name": "Condition1", "likelihood": 60}},
        {{"name": "Condition2", "likelihood": 30}},
        {{"name": "Condition3", "likelihood": 10}}
    ],
    "risk_level": "low/moderate/high",
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2"
    ],
    "reference": "Medical guideline reference"
}}

Patient message: """
{user_message}
"""

Respond in the same language as the user (Bahasa Indonesia or English).
'''

def get_gemini_response(user_message: str, api_key: str) -> Dict:
    prompt = PROMPT_TEMPLATE.format(user_message=user_message)
    messages = [
        {"role": "user", "parts": [prompt]}
    ]
    import json
    try:
        response = call_gemini_api(messages, api_key)
        # Try to extract JSON from response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(response[start:end])
        else:
            return {
                "symptoms": ["unclear symptoms"],
                "conditions": [{"name": "Unknown", "likelihood": 100}],
                "risk_level": "low",
                "recommendations": ["Please clarify your symptoms"],
                "reference": "N/A"
            }
    except Exception as e:
        return {
            "symptoms": ["error"],
            "conditions": [{"name": "Error", "likelihood": 100}],
            "risk_level": "low", 
            "recommendations": [f"Failed to get response: {str(e)}"],
            "reference": "N/A"
        }

def create_likelihood_chart(conditions):
    """Create a bar chart showing condition likelihoods"""
    if not conditions:
        return None
    
    df = pd.DataFrame(conditions)
    
    # Truncate long condition names for better display
    df['short_name'] = df['name'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
    
    fig = px.bar(
        df, 
        x='likelihood', 
        y='short_name',
        orientation='h',
        title='Condition Likelihood (%)',
        labels={'likelihood': 'Likelihood (%)', 'short_name': 'Condition'},
        color='likelihood',
        color_continuous_scale='RdYlBu_r',
        text='likelihood'  # Show percentages on bars
    )
    
    # Update text formatting and layout
    fig.update_traces(
        texttemplate='%{text}%',
        textposition='outside',
        textfont_size=12
    )
    
    fig.update_layout(
        height=max(250, len(conditions) * 60),  # Adjust height based on number of conditions
        showlegend=False,
        xaxis=dict(range=[0, max(df['likelihood']) * 1.2])  # Add space for text
    )
    
    return fig

def get_risk_badge_color(risk_level):
    """Get color for risk level badge"""
    colors = {
        'low': 'green',
        'moderate': 'orange', 
        'high': 'red'
    }
    return colors.get(risk_level.lower(), 'gray')

def display_triage_results(triage_data):
    """Display triage results with enhanced UI"""
    
    # Risk Level Badge (translate risk levels)
    risk_level = triage_data.get('risk_level', 'low')
    risk_translations = {
        'low': 'RENDAH',
        'moderate': 'SEDANG', 
        'high': 'TINGGI'
    }
    risk_level_id = risk_translations.get(risk_level.lower(), risk_level.upper())
    risk_color = get_risk_badge_color(risk_level)
    
    st.markdown(f"""
    <div style="background-color: {risk_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
        <h3>Tingkat Risiko: {risk_level_id}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Extracted Symptoms
    symptoms = triage_data.get('symptoms', [])
    if symptoms:
        st.subheader("🔍 Gejala yang Teridentifikasi:")
        # Display symptoms side by side using markdown with inline format
        symptoms_text = " • ".join(symptoms)
        st.markdown(f"{symptoms_text}")
    
    # Likelihood Chart
    conditions = triage_data.get('conditions', [])
    if conditions:
        st.subheader("📊 Analisis Kondisi:")
        
        # Show conditions as text list with percentages (cleaner for long names)
        for condition in conditions:
            percentage = condition['likelihood']
            name = condition['name']
            # Create a visual bar using markdown on separate line
            bar_length = int(percentage / 5)  # Scale bar length
            bar = "🟩" * bar_length + "⬜" * (20 - bar_length)
            
            # Display name and percentage on first line, bar on second line
            st.markdown(f"**{name}**: {percentage}%")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{bar}", unsafe_allow_html=True)
        
        # Also show the chart for visual appeal (with truncated names)
        chart = create_likelihood_chart(conditions)
        if chart:
            with st.expander("📈 Lihat Grafik"):
                st.plotly_chart(chart, use_container_width=True)
    
    # Recommendations
    recommendations = triage_data.get('recommendations', [])
    if recommendations:
        st.subheader("💡 Rekomendasi:")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. ✅ {rec}")
    
    return triage_data

def get_related_symptoms(condition_name):
    """Get related symptoms based on the diagnosed condition"""
    # Dictionary of common related symptoms for different conditions
    symptom_groups = {
        "dengue": ["skin rash", "joint pain", "eye pain", "bleeding gums"],
        "influenza": ["muscle aches", "runny nose", "sore throat", "fatigue"],
        "flu": ["muscle aches", "runny nose", "sore throat", "fatigue"],
        "viral": ["runny nose", "sore throat", "body aches", "fatigue"],
        "fever": ["chills", "sweating", "weakness", "loss of appetite"],
        "respiratory": ["runny nose", "sore throat", "sneezing", "congestion"],
        "gastrointestinal": ["stomach cramps", "loss of appetite", "dehydration", "weakness"],
        "diarrhea": ["stomach cramps", "dehydration", "loss of appetite", "weakness"],
        "typhoid": ["rose spots", "enlarged spleen", "weakness", "loss of appetite"],
        "chest pain": ["shortness of breath", "sweating", "dizziness", "arm pain"],
        "headache": ["sensitivity to light", "nausea", "neck stiffness", "dizziness"],
        "cold": ["runny nose", "sneezing", "sore throat", "congestion"],
        "measles": ["skin rash", "eye irritation", "sensitivity to light", "white spots in mouth"],
        "chickenpox": ["skin rash", "blisters", "itching", "fatigue"]
    }
    
    # Find matching symptoms based on keywords in condition name
    related = []
    condition_lower = condition_name.lower()
    
    for key, symptoms in symptom_groups.items():
        if key in condition_lower:
            related.extend(symptoms)
    
    # Remove duplicates and limit to 3 most relevant
    related = list(dict.fromkeys(related))[:3]
    
    return related

def extract_symptoms_simple(user_input):
    """Simple symptom extraction from user input without full AI analysis"""
    # Common symptom keywords to look for (expanded with Indonesian terms)
    symptom_keywords = {
        "demam": ["fever", "demam", "hot", "panas", "meriang"],
        "sakit kepala": ["headache", "sakit kepala", "head pain", "pusing", "pening"],
        "batuk": ["cough", "batuk", "coughing", "batuk kering"],
        "batuk berdahak": ["batuk berdahak", "berdahak", "productive cough", "wet cough", "batuk berlendir"],
        "mual": ["nausea", "mual", "feel sick", "queasy", "eneg"],
        "muntah": ["vomit", "muntah", "throw up", "throwing up"],
        "diare": ["diarrhea", "diare", "loose stool", "watery stool", "mencret"],
        "sakit perut": ["abdominal pain", "stomach pain", "sakit perut", "belly pain", "perut sakit", "nyeri perut"],
        "nyeri dada": ["chest pain", "sakit dada", "chest hurt", "nyeri dada"],
        "sesak nafas": ["shortness of breath", "sesak nafas", "difficulty breathing", "hard to breathe", "susah nafas", "sulit bernapas", "sulit bernafas"],
        "kelelahan": ["tired", "fatigue", "lelah", "weakness", "lemah", "lemas"],
        "berkeringat": ["sweating", "berkeringat", "keringat dingin", "berkeringat dingin", "cold sweat"],
        "sakit tenggorokan": ["sore throat", "sakit tenggorokan", "throat pain"],
        "pilek": ["runny nose", "pilek", "nasal congestion", "hidung tersumbat"],
        "nyeri otot": ["body aches", "muscle pain", "nyeri otot", "pegal", "badan pegal"],
        "tidak nafsu makan": ["loss of appetite", "tidak nafsu makan", "no appetite", "sulit makan", "susah makan", "hilang nafsu makan"]
    }
    
    user_input_lower = user_input.lower()
    extracted = []
    
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                extracted.append(symptom)
                break  # Found one match for this symptom, move to next
    
    # Remove duplicates and return
    return list(dict.fromkeys(extracted))

def get_related_symptoms_simple(extracted_symptoms):
    """Get related symptoms based on simple symptom relationships"""
    # Simple symptom relationship mapping (Indonesian terms)
    symptom_relationships = {
        "demam": ["menggigil", "berkeringat", "nyeri otot", "sakit kepala"],
        "sakit kepala": ["mual", "sensitif cahaya", "kaku leher"],
        "mual": ["muntah", "tidak nafsu makan", "pusing"],
        "batuk": ["sakit tenggorokan", "nyeri dada", "sesak nafas"],
        "batuk berdahak": ["sakit tenggorokan", "nyeri dada", "sesak nafas", "demam"],
        "muntah": ["dehidrasi", "lemas", "kram perut"],
        "sakit perut": ["mual", "muntah", "tidak nafsu makan"],
        "sesak nafas": ["nyeri dada", "pusing", "berkeringat"],
        "nyeri dada": ["sesak nafas", "nyeri lengan", "berkeringat"],
        "diare": ["kram perut", "dehidrasi", "lemas"],
        "kelelahan": ["lemas", "pusing", "tidak nafsu makan"],
        "berkeringat": ["menggigil", "demam", "lemas", "pusing"],
        "sakit tenggorokan": ["pilek", "batuk", "kelenjar bengkak"],
        "pilek": ["bersin", "hidung tersumbat", "sakit tenggorokan"]
    }
    
    related = []
    
    # Get related symptoms for each extracted symptom
    for symptom in extracted_symptoms:
        if symptom in symptom_relationships:
            related.extend(symptom_relationships[symptom])
    
    # Remove symptoms they already have and duplicates
    extracted_lower = [s.lower() for s in extracted_symptoms]
    related = [s for s in related if s.lower() not in extracted_lower]
    related = list(dict.fromkeys(related))[:5]  # Remove duplicates and limit to 5
    
    return related

def get_related_symptoms_from_extraction(extracted_symptoms, condition_name):
    """Get related symptoms based on extracted symptoms and diagnosed condition"""
    # Dictionary of common related symptoms for different conditions
    symptom_groups = {
        "dengue": ["skin rash", "joint pain", "eye pain", "bleeding gums", "abdominal pain"],
        "influenza": ["muscle aches", "runny nose", "sore throat", "fatigue", "chills"],
        "flu": ["muscle aches", "runny nose", "sore throat", "fatigue", "chills"],
        "viral": ["runny nose", "sore throat", "body aches", "fatigue", "congestion"],
        "fever": ["chills", "sweating", "weakness", "loss of appetite", "body aches"],
        "respiratory": ["runny nose", "sore throat", "sneezing", "congestion", "chest tightness"],
        "gastrointestinal": ["stomach cramps", "loss of appetite", "dehydration", "weakness"],
        "diarrhea": ["stomach cramps", "dehydration", "loss of appetite", "weakness"],
        "typhoid": ["rose spots", "enlarged spleen", "weakness", "loss of appetite"],
        "chest pain": ["shortness of breath", "sweating", "dizziness", "arm pain"],
        "headache": ["sensitivity to light", "nausea", "neck stiffness", "dizziness"],
        "cold": ["runny nose", "sneezing", "sore throat", "congestion"],
        "measles": ["skin rash", "eye irritation", "sensitivity to light", "white spots in mouth"],
        "chickenpox": ["skin rash", "blisters", "itching", "fatigue"]
    }
    
    # Additional symptom relationships based on what they already have
    symptom_relationships = {
        "fever": ["chills", "sweating", "body aches", "headache"],
        "headache": ["nausea", "sensitivity to light", "neck stiffness"],
        "nausea": ["vomiting", "loss of appetite", "dizziness"],
        "cough": ["sore throat", "chest pain", "shortness of breath"],
        "vomiting": ["dehydration", "weakness", "stomach cramps"],
        "abdominal pain": ["nausea", "vomiting", "loss of appetite"],
        "breathing difficulty": ["chest pain", "dizziness", "sweating"],
        "chest pain": ["shortness of breath", "arm pain", "sweating"]
    }
    
    related = []
    condition_lower = condition_name.lower()
    
    # Get related symptoms based on condition
    for key, symptoms in symptom_groups.items():
        if key in condition_lower:
            related.extend(symptoms)
    
    # Get related symptoms based on what they already reported
    for symptom in extracted_symptoms:
        symptom_lower = symptom.lower()
        for key, related_symp in symptom_relationships.items():
            if key in symptom_lower:
                related.extend(related_symp)
    
    # Remove symptoms they already mentioned and duplicates
    extracted_lower = [s.lower() for s in extracted_symptoms]
    related = [s for s in related if not any(s.lower() in ext.lower() or ext.lower() in s.lower() for ext in extracted_lower)]
    related = list(dict.fromkeys(related))[:3]  # Remove duplicates and limit to 3
    
    return related

# Load guidelines from local folder
@st.cache_resource
def load_guidelines():
    guideline_folder = os.path.join(os.path.dirname(__file__), "guidelines")
    docs = []
    
    # Load all markdown files
    md_files = glob.glob(os.path.join(guideline_folder, "*.md"))
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append({
                    'content': content,
                    'source': os.path.basename(file_path)
                })
        except Exception as e:
            st.warning(f"Could not load {file_path}: {e}")
    
    return docs

# Streamlit UI
def main():
    st.set_page_config(page_title="Patient Symptom Triage Chatbot", page_icon="💬")
    
    # Create title row with restart button
    title_col, button_col = st.columns([4, 1])
    
    with title_col:
        st.title("💬 Chatbot Triase Gejala Pasien")
        st.markdown("Chat dalam Bahasa Indonesia atau English. Jelaskan gejala Anda dan dapatkan saran triase.")
    
    with button_col:
        def clear_conversation():
            st.session_state.chat_history = []
            st.session_state.selected_symptom = None
            st.session_state.symptom_collection_mode = False
            st.session_state.collected_symptoms = []
            st.session_state.selected_additional_symptoms = []
            st.session_state.trigger_analysis = False
            if "prev_question_timestamp" in st.session_state:
                st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
        
        # Always show restart button if there's any interaction
        if (st.session_state.get("chat_history", []) or 
            st.session_state.get("collected_symptoms", []) or
            st.session_state.get("selected_symptom") or
            st.session_state.get("symptom_collection_mode", False)):
            st.button(
                "🔄 Restart",
                type="secondary",
                on_click=clear_conversation,
                use_container_width=True,
                help="Mulai percakapan baru"
            )

    # Secure API Key Management
    def get_api_key():
        """Get API key from multiple sources with fallback options"""
        # Method 1: Try Streamlit secrets (for local development)
        try:
            if "api_keys" in st.secrets:
                return st.secrets["api_keys"]["gemini_api_key"]
            elif "gemini_api_key" in st.secrets:
                return st.secrets["gemini_api_key"]
        except Exception:
            pass
        
        # Method 2: Try environment variable
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
        
        # Method 3: Try session state (user input)
        if "user_api_key" in st.session_state and st.session_state.user_api_key:
            return st.session_state.user_api_key
        
        return None

    # Get API key
    api_key = get_api_key()
    
    # Show API key input if not found
    if not api_key:
        st.warning("⚠️ Gemini API key tidak ditemukan!")
        st.info("Masukkan API key Anda di bawah ini:")
        
        with st.expander("🔑 Pengaturan API Key", expanded=True):
            user_api_key = st.text_input(
                "Gemini API Key:",
                type="password",
                help="Dapatkan API key gratis di https://makersuite.google.com/app/apikey",
                placeholder="Masukkan API key Anda di sini..."
            )
            if st.button("Simpan API Key", type="primary"):
                if user_api_key:
                    st.session_state.user_api_key = user_api_key
                    st.success("✅ API Key berhasil disimpan!")
                    st.rerun()
                else:
                    st.error("❌ Silakan masukkan API key yang valid.")
            
            st.markdown("""
            **Cara mendapatkan API Key:**
            1. Kunjungi [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Login dengan akun Google Anda
            3. Klik "Create API Key"
            4. Copy dan paste API key ke form di atas
            """)
        
        st.stop()  # Stop execution until API key is provided
    
    # Sidebar for Guidelines
    with st.sidebar:
        st.header("📋 Panduan Medis")
        st.markdown("Materi referensi untuk tenaga kesehatan:")
        
        # Guidelines with web links
        guidelines_links = {
            "🦟 Panduan Demam Berdarah": "https://www.who.int/publications/i/item/9789241547871",
            "🤧 Panduan Influenza": "https://www.cdc.gov/flu/professionals/diagnosis/clinician_guidance_ridt.htm",
            "💧 Panduan Diare": "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease",
            "🤒 Panduan Tifus": "https://www.who.int/news-room/fact-sheets/detail/typhoid",
            "💊 Panduan Nyeri Dada": "https://www.heart.org/en/health-topics/consumer-healthcare/what-is-cardiovascular-disease/angina-chest-pain"
        }
        
        for guideline_name, link in guidelines_links.items():
            st.markdown(f"- [{guideline_name}]({link})")
        
        st.markdown("---")
        st.markdown("**Catatan:** Ini adalah tautan referensi eksternal. Selalu konsultasi dengan tenaga kesehatan untuk diagnosis dan pengobatan yang tepat.")
    
    # Predefined symptom examples
    symptom_examples = [
        "🤒 Saya demam dan sakit kepala selama 3 hari",
        "🤢 Saya mengalami mual, muntah, dan sakit perut", 
        "💨 Saya sulit bernapas dan nyeri dada"
    ]

    def get_relevant_guideline(conditions, symptoms):
        """Get relevant guideline based on conditions and symptoms"""
        docs = load_guidelines()
        if not docs:
            return "No guidelines available. Please consult with a healthcare professional for proper medical advice."
        
        # Simple keyword matching for demo (replace with better retrieval in production)
        search_terms = [cond["name"].lower() for cond in conditions] + [sym.lower() for sym in symptoms]
        
        best_match = None
        best_score = 0
        
        for doc in docs:
            content_lower = doc['content'].lower()
            score = sum(1 for term in search_terms if term in content_lower)
            if score > best_score:
                best_score = score
                best_match = doc
        
        if best_match:
            content = best_match['content']
            return content[:1000] + "..." if len(content) > 1000 else content
        else:
            # Return the first guideline as fallback
            fallback = docs[0]['content']
            return fallback[:1000] + "..." if len(fallback) > 1000 else fallback

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "collected_symptoms" not in st.session_state:
        st.session_state.collected_symptoms = []
    if "selected_additional_symptoms" not in st.session_state:
        st.session_state.selected_additional_symptoms = []
    if "symptom_collection_mode" not in st.session_state:
        st.session_state.symptom_collection_mode = False
    
    # Only show symptom examples if chat history is empty
    if not st.session_state.chat_history:
        st.markdown("### 💡 Pilih dari contoh cepat:")
        
        # Create columns for symptom buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(symptom_examples[0], key="symptom1", use_container_width=True):
                st.session_state.selected_symptom = symptom_examples[0]
                st.rerun()
        
        with col2:
            if st.button(symptom_examples[1], key="symptom2", use_container_width=True):
                st.session_state.selected_symptom = symptom_examples[1]
                st.rerun()
        
        with col3:
            if st.button(symptom_examples[2], key="symptom3", use_container_width=True):
                st.session_state.selected_symptom = symptom_examples[2]
                st.rerun()

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
            # Display follow-up questions if this is a follow-up message (simplified for history)
            if chat.get("follow_up"):
                extracted_symptoms = chat.get("extracted_symptoms", [])
                
                if extracted_symptoms:
                    st.markdown(f"**Gejala yang teridentifikasi:** {', '.join(extracted_symptoms)}")
            
            # Display final triage analysis only for final analysis
            if chat.get("final_analysis") and chat.get("triage"):
                display_triage_results(chat["triage"])
                
                # Add next steps after analysis
                st.markdown("---")
                st.markdown("## 🔄 Langkah Selanjutnya")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🆕 Analisis Baru", key="new_analysis", use_container_width=True, type="primary"):
                        # Clear all session state for new analysis
                        st.session_state.chat_history = []
                        st.session_state.selected_symptom = None
                        st.session_state.symptom_collection_mode = False
                        st.session_state.collected_symptoms = []
                        st.session_state.selected_additional_symptoms = []
                        st.session_state.trigger_analysis = False
                        st.rerun()
                
                with col2:
                    if st.button("📋 Salin Hasil", key="copy_results", use_container_width=True):
                        # Show copyable text version of results
                        triage_data = chat.get("triage", {})
                        result_text = f"""
HASIL ANALISIS TRIASE MEDIS

Tingkat Risiko: {triage_data.get('risk_level', 'Unknown').upper()}

Gejala yang Teridentifikasi:
{' • '.join(triage_data.get('symptoms', []))}

Analisis Kondisi:
"""
                        for condition in triage_data.get('conditions', []):
                            result_text += f"• {condition['name']}: {condition['likelihood']}%\n"
                        
                        result_text += f"""
Rekomendasi:
"""
                        for i, rec in enumerate(triage_data.get('recommendations', []), 1):
                            result_text += f"{i}. {rec}\n"
                        
                        st.text_area("Salin teks di bawah ini:", result_text, height=200)
                
                with col3:
                    if st.button("ℹ️ Info Panduan", key="show_guidelines", use_container_width=True):
                        st.info("Silakan lihat panduan medis di sidebar untuk informasi lebih lanjut tentang kondisi dan gejala yang dialami.")
                
                st.markdown("---")
                st.markdown("""
                <div style="background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107;">
                    <strong>⚠️ Penting:</strong> Hasil analisis ini hanya untuk panduan awal. 
                    Untuk diagnosis dan pengobatan yang akurat, selalu konsultasikan dengan tenaga kesehatan profesional.
                </div>
                """, unsafe_allow_html=True)




    # Chat input - always visible
    chat_input = st.chat_input("Jelaskan gejala Anda...")
    
    # Handle selected symptom from buttons or chat input
    user_input = None
    if "selected_symptom" in st.session_state:
        user_input = st.session_state.selected_symptom
        del st.session_state.selected_symptom  # Clear the selection
    elif chat_input:
        user_input = chat_input
    
    if user_input:
        # Handle continuation of symptom collection
        if user_input == "continue_symptom_collection":
            # Get all previously mentioned symptoms using simple extraction
            all_previous_symptoms = []
            for symptom_text in st.session_state.collected_symptoms:
                all_previous_symptoms.extend(extract_symptoms_simple(symptom_text))
            
            with st.spinner("Mencari gejala terkait lainnya..."):
                # Get related symptoms based on all collected symptoms
                related_symptoms = get_related_symptoms_simple(all_previous_symptoms)
            
            # Show follow-up questions for additional symptoms
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"Bagus! Sekarang mari saya tanyakan tentang gejala terkait lainnya:",
                "follow_up": True,
                "related_symptoms": related_symptoms,
                "extracted_symptoms": all_previous_symptoms
            })
            st.rerun()
            return
        # Initialize symptom collection session state
        if "symptom_collection_mode" not in st.session_state:
            st.session_state.symptom_collection_mode = True
            st.session_state.collected_symptoms = []
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add current symptoms to collection
        st.session_state.collected_symptoms.append(user_input)
        
        # Quick symptom extraction (without full analysis)
        with st.spinner("Mengekstrak gejala..."):
            extracted_symptoms = extract_symptoms_simple(user_input)
            
            # Get related symptoms based on extracted symptoms only (no condition analysis needed yet)
            related_symptoms = get_related_symptoms_simple(extracted_symptoms)
        
        # Show follow-up questions first
        if extracted_symptoms:
            assistant_message = f"Saya memahami Anda mengalami: **{', '.join(extracted_symptoms)}**. Mari saya tanyakan tentang gejala terkait lainnya:"
        else:
            assistant_message = "Saya telah menerima pesan Anda. Mari saya tanyakan tentang gejala terkait lainnya:"
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": assistant_message,
            "follow_up": True,
            "related_symptoms": related_symptoms,
            "extracted_symptoms": extracted_symptoms
        })
        
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

    # Show single interactive symptom selection interface
    # Find the most recent follow-up message
    latest_followup = None
    related_symptoms = []
    if st.session_state.get("chat_history", []):
        for chat in reversed(st.session_state.chat_history):
            if chat.get("follow_up", False):
                latest_followup = chat
                related_symptoms = chat.get("related_symptoms", [])
                break
    
    # Check if analysis is finished
    analysis_finished = any(chat.get("final_analysis", False) for chat in st.session_state.get("chat_history", []))
    
    # Show interface if we have related symptoms and haven't finished analysis
    if related_symptoms and not analysis_finished:
        st.markdown("---")
        st.markdown("## 🤔 Apakah Anda mengalami gejala tambahan berikut?")
        st.markdown("*Gunakan antarmuka di bawah ini untuk memilih gejala tambahan.*")
        
        # Initialize selected additional symptoms in session state
        if "selected_additional_symptoms" not in st.session_state:
            st.session_state.selected_additional_symptoms = []
        
        # Create multiselect for additional symptoms
        selected_symptoms = st.multiselect(
            "Pilih gejala tambahan yang Anda alami:",
            options=related_symptoms,
            default=st.session_state.selected_additional_symptoms,
            key=f"current_additional_symptoms_{len(st.session_state.get('chat_history', []))}_{hash(str(related_symptoms))}"
        )
        
        # Update session state
        st.session_state.selected_additional_symptoms = selected_symptoms
        
        # Show selected symptoms as tags
        if selected_symptoms:
            st.markdown("**Gejala tambahan yang dipilih:**")
            tags_html = ""
            for symptom in selected_symptoms:
                tags_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 12px; display: inline-block;">🏷️ {symptom}</span>'
            st.markdown(tags_html, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Tambah gejala terpilih", key=f"current_add_selected_{len(st.session_state.get('chat_history', []))}", use_container_width=True):
                if st.session_state.selected_additional_symptoms:
                    # Add selected symptoms to collected symptoms
                    additional_symptoms_text = f"Saya juga mengalami: {', '.join(st.session_state.selected_additional_symptoms)}"
                    st.session_state.collected_symptoms.append(additional_symptoms_text)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": additional_symptoms_text})
                    
                    # Clear selected symptoms for next round
                    st.session_state.selected_additional_symptoms = []
                    
                    # Continue with more symptom collection
                    st.session_state.selected_symptom = "continue_symptom_collection"
                    st.rerun()
                else:
                    st.warning("Silakan pilih setidaknya satu gejala terlebih dahulu.")
        
        with col2:
            if st.button("✅ Selesai, analisis sekarang", key=f"current_done_{len(st.session_state.get('chat_history', []))}", type="primary", use_container_width=True):
                # Add any remaining selected symptoms before analysis
                if st.session_state.get("selected_additional_symptoms", []):
                    additional_symptoms_text = f"Saya juga mengalami: {', '.join(st.session_state.selected_additional_symptoms)}"
                    st.session_state.collected_symptoms.append(additional_symptoms_text)
                    st.session_state.chat_history.append({"role": "user", "content": additional_symptoms_text})
                
                # Ensure we have collected symptoms
                collected = st.session_state.get("collected_symptoms", [])
                if not collected:
                    st.error("Tidak ada gejala yang terkumpul. Silakan coba lagi.")
                    return
                
                # Set a flag to trigger analysis
                st.session_state.trigger_analysis = True
                st.rerun()    # Handle analysis trigger (must be outside the user_input block)
    if st.session_state.get("trigger_analysis", False):
        # Clear the trigger flag first
        st.session_state.trigger_analysis = False
        
        # Debug: Show what symptoms we have
        collected = st.session_state.get("collected_symptoms", [])

        
        # Perform final analysis with all collected symptoms
        if collected:
            all_symptoms = " dan ".join(collected)
            final_input = f"Ringkasan gejala lengkap: {all_symptoms}"
            
            with st.spinner("Melakukan analisis komprehensif..."):
                final_triage = get_gemini_response(final_input, api_key)
            
            # Add final analysis to chat
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "## 🏥 Analisis Triase Medis Lengkap", 
                "triage": final_triage,
                "final_analysis": True
            })
            
            # Reset symptom collection mode
            st.session_state.symptom_collection_mode = False
            st.session_state.collected_symptoms = []
            st.session_state.selected_additional_symptoms = []
            st.rerun()

    # Initialize timestamp tracking
    if "prev_question_timestamp" not in st.session_state:
        st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

if __name__ == "__main__":
    main()
