import streamlit as st
import google.generativeai as genai
import json
import os
from typing import Dict, List, Optional
import plotly.express as px
import pandas as pd
import time
import datetime

def list_available_models(api_key: str):
    """Debug function to list available models"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        return available_models
    except Exception as e:
        return [f"Error listing models: {str(e)}"]

# Check for hybrid functionality
try:
    from hybrid_retrieval import HybridMedicalRetriever
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

st.set_page_config(
    page_title="Patient Symptom Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure page settings
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def call_gemini_api(messages: List[Dict[str, str]], api_key: str) -> Dict:
    """Call Gemini API with structured messages"""
    genai.configure(api_key=api_key)
    
    # Try different model names with proper API versions
    model_configs = [
        'gemini-1.5-flash-001',
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-pro-001', 
        'gemini-pro-001',
        'gemini-1.5-flash'
    ]
    
    last_error = None
    for model_name in model_configs:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(messages[0]['content'])
            return response.text
        except Exception as e:
            last_error = e
            if "not found" in str(e).lower():
                continue  # Try next model
            else:
                break  # Different error, stop trying
    
    # If all models failed, raise the last error
    raise last_error

def get_gemini_response_hybrid(user_message: str, api_key: str, exa_api_key: Optional[str] = None) -> Dict:
    """Get response using hybrid approach with local guidelines and web research"""
    try:
        # Return basic mode if hybrid is not available
        if not HYBRID_AVAILABLE:
            return get_gemini_response_basic(user_message, api_key)
            
        # Initialize hybrid retriever
        retriever = HybridMedicalRetriever(gemini_api_key=api_key, exa_api_key=exa_api_key)
        
        # Prepare hybrid message with context from local guidelines and web research
        context_data = retriever.hybrid_search(user_message, [], [])
        
        hybrid_prompt = f"""
        Sebagai dokter AI dengan akses ke pedoman medis terkini dan penelitian medis terpercaya, lakukan analisis komprehensif:

        GEJALA PASIEN: {user_message}

        KONTEKS MEDIS DARI DATABASE:
        {context_data.get('context', 'Tidak ada konteks tambahan tersedia')}

        SUMBER MEDIS TERPERCAYA: {len(context_data.get('sources', []))} pedoman dan penelitian

        INSTRUKSI ANALISIS:
        1. Identifikasi 3-5 kondisi medis dengan likelihood realistis
        2. Berikan deskripsi detail setiap kondisi
        3. Tentukan tingkat urgensi berdasarkan evidens medis
        4. Sertakan rekomendasi spesifik dan actionable
        5. Identifikasi red flags berdasarkan pedoman medis

        Format JSON response:
        {{
            "conditions": [
                {{"name": "Influenza/Flu", "likelihood": 78, "symptoms": ["demam tinggi", "sakit kepala", "nyeri otot", "kelelahan"], "description": "Infeksi virus dengan gejala sistemik yang cocok dengan presentasi pasien. Likelihood tinggi berdasarkan kombinasi demam, sakit kepala, dan nyeri otot."}},
                {{"name": "COVID-19", "likelihood": 65, "symptoms": ["demam", "sakit kepala", "kelelahan"], "description": "Infeksi SARS-CoV-2 dengan gejala mirip flu. Perlu pertimbangan karena masih dalam sirkulasi komunitas."}},
                {{"name": "Dengue Fever", "likelihood": 45, "symptoms": ["demam tinggi", "sakit kepala", "nyeri otot"], "description": "Kemungkinan dengue terutama jika ada riwayat gigitan nyamuk atau endemik di area tersebut."}}
            ],
            "triage": {{
                "urgency": "medium",
                "priority": 3,
                "recommendation": "Konsultasi dokter dalam 24-48 jam untuk evaluasi dan konfirmasi diagnosis",
                "reasoning": "Kombinasi demam tinggi dan gejala sistemik memerlukan evaluasi medis untuk membedakan antara infeksi virus dan kemungkinan kondisi yang memerlukan perawatan spesifik"
            }},
            "recommendations": [
                "Istirahat total dan hidrasi adekuat (8-10 gelas air per hari)",
                "Monitor suhu tubuh setiap 4-6 jam",
                "Konsumsi paracetamol 500mg setiap 6-8 jam untuk demam",
                "Isolasi mandiri jika suspek infeksi menular",
                "Konsumsi makanan bergizi dan mudah dicerna"
            ],
            "red_flags": [
                "Demam di atas 39°C yang persisten lebih dari 3 hari",
                "Sesak napas atau kesulitan bernapas",
                "Penurunan kesadaran atau confusion",
                "Tanda-tanda dehidrasi berat"
            ],
            "follow_up": "Konsultasi dokter dalam 24-48 jam. Segera ke UGD jika mengalami red flags atau gejala memburuk"
        }}

        PENTING: Response harus JSON valid tanpa ```json atau markdown formatting apapun.
        """

        # Generate response
        genai.configure(api_key=api_key)
        
        # Try different model names with proper API versions
        model_configs = [
            'gemini-1.5-flash-001',
            'gemini-1.5-pro-001', 
            'gemini-pro-001',
            'gemini-1.5-flash'
        ]
        
        last_error = None
        for model_name in model_configs:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(hybrid_prompt)
                break
            except Exception as e:
                last_error = e
                if "not found" in str(e).lower():
                    continue  # Try next model
                else:
                    break  # Different error, stop trying
        else:
            # If all models failed, raise the last error
            raise last_error
        
        try:
            # Try to parse as JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            
            # Add source information to the result
            result['sources_used'] = {
                'total_sources': len(context_data.get('sources', [])),
                'local_guidelines': len([s for s in context_data.get('sources', []) if s.get('type') == 'local']),
                'web_research': len([s for s in context_data.get('sources', []) if s.get('type') == 'web']),
                'sources': context_data.get('sources', [])
            }
            
            return result
            
        except json.JSONDecodeError as e:
            # Silently fallback without interrupting user experience
            if st.session_state.get('debug_mode', False):
                st.warning(f"JSON parsing failed in hybrid mode, using basic analysis.")
            return get_gemini_response_basic(user_message, api_key)
            
    except Exception as e:
        # Silently fallback to basic analysis without showing error to user during progress
        # Only log the error for debugging if needed
        if st.session_state.get('debug_mode', False):
            st.warning(f"Hybrid mode failed, using basic analysis: {str(e)}")
        return get_gemini_response_basic(user_message, api_key)

def get_gemini_response_basic(user_message: str, api_key: str) -> Dict:
    """Basic Gemini response without hybrid features"""
    prompt = f"""
    Sebagai dokter AI yang berpengalaman, lakukan analisis mendalam terhadap gejala pasien berikut:

    GEJALA PASIEN: {user_message}

    Instruksi:
    1. Identifikasi minimal 3-5 kondisi medis yang mungkin
    2. Berikan persentase likelihood yang realistis (tinggi untuk kondisi yang sangat mungkin)
    3. Sertakan gejala terkait untuk setiap kondisi
    4. Berikan rekomendasi spesifik dan actionable
    5. Tentukan tingkat urgensi berdasarkan gejala
    6. Identifikasi red flags jika ada

    Response dalam format JSON berikut:
    {{
        "conditions": [
            {{"name": "Kondisi Medis Spesifik", "likelihood": 75, "symptoms": ["demam tinggi", "sakit kepala", "nyeri otot"], "description": "Penjelasan detail kondisi dan mengapa kemungkinannya tinggi berdasarkan gejala"}},
            {{"name": "Kondisi Alternatif", "likelihood": 45, "symptoms": ["gejala1", "gejala2"], "description": "Penjelasan alternatif kondisi"}},
            {{"name": "Kondisi Lain", "likelihood": 30, "symptoms": ["gejala3"], "description": "Kemungkinan lain yang perlu dipertimbangkan"}}
        ],
        "triage": {{
            "urgency": "medium",
            "priority": 3,
            "recommendation": "Konsultasi dengan dokter dalam 24-48 jam untuk evaluasi lebih lanjut",
            "reasoning": "Berdasarkan kombinasi gejala yang dialami, diperlukan evaluasi medis untuk konfirmasi diagnosis"
        }},
        "recommendations": [
            "Istirahat yang cukup dan minum banyak air",
            "Monitor suhu tubuh setiap 4 jam",
            "Konsumsi paracetamol untuk demam jika diperlukan",
            "Hindari aktivitas berat hingga gejala membaik",
            "Konsultasi dokter jika gejala memburuk"
        ],
        "red_flags": [
            "Demam di atas 39°C yang tidak turun dengan obat",
            "Kesulitan bernapas atau sesak napas",
            "Nyeri dada yang hebat"
        ],
        "follow_up": "Konsultasi dokter dalam 24-48 jam. Jika mengalami red flags, segera ke UGD"
    }}

    PENTING: Berikan hanya JSON valid tanpa ```json atau ``` formatting.
    """

    try:
        response = call_gemini_api([{"content": prompt}], api_key)
        # Clean and parse JSON
        response_text = response.strip()
        
        # Try to extract JSON from various formats
        if response_text.startswith('```json'):
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_text = response_text[start:end]
        elif response_text.startswith('```'):
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_text = response_text[start:end]
        elif response_text.startswith('{') and response_text.endswith('}'):
            # Direct JSON response
            json_text = response_text
        else:
            # Try to find JSON within the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_text = response_text[start:end]
            else:
                raise json.JSONDecodeError("No JSON found in response", response_text, 0)
        
        return json.loads(json_text)
        
    except json.JSONDecodeError as e:
        # Only show debug info if debug mode is enabled
        if st.session_state.get('debug_mode', False):
            st.error(f"JSON parsing failed: {str(e)}")
            st.text_area("Raw AI Response (for debugging):", response_text, height=200)
        
        # Return a more graceful fallback response
        return {
            "conditions": [
                {"name": "Analisis Medis", "likelihood": 70, "symptoms": ["Berdasarkan gejala yang disebutkan"], "description": "Analisis berdasarkan gejala yang telah dijelaskan. Memerlukan evaluasi medis lebih lanjut untuk diagnosis yang akurat."}
            ],
            "triage": {"urgency": "medium", "priority": 3, "recommendation": "Konsultasi dengan dokter dalam 24-48 jam", "reasoning": "Berdasarkan analisis gejala yang dijelaskan"},
            "recommendations": ["Istirahat yang cukup", "Monitor perkembangan gejala", "Konsultasi dokter jika gejala memburuk"],
            "red_flags": ["Gejala yang memburuk secara signifikan", "Demam tinggi yang persisten"],
            "follow_up": "Konsultasi dokter dalam 24-48 jam atau segera jika gejala memburuk"
        }
    except Exception as e:
        # Only show system errors in debug mode
        if st.session_state.get('debug_mode', False):
            st.error(f"System error: {str(e)}")
        
        return {
            "conditions": [
                {"name": "Evaluasi Medis Diperlukan", "likelihood": 60, "symptoms": ["Gejala yang dilaporkan"], "description": "Diperlukan evaluasi medis lebih lanjut untuk analisis yang komprehensif."}
            ],
            "triage": {"urgency": "medium", "priority": 3, "recommendation": "Konsultasi dengan dokter", "reasoning": "Evaluasi medis diperlukan"},
            "recommendations": ["Konsultasi dengan tenaga medis profesional"],
            "red_flags": [],
            "follow_up": "Segera konsultasi dengan tenaga medis"
        }

def get_gemini_response(user_message: str, api_key: str, exa_api_key: Optional[str] = None) -> Dict:
    """Route to appropriate response method"""
    if HYBRID_AVAILABLE and st.session_state.get('use_hybrid_mode', True):
        return get_gemini_response_hybrid(user_message, api_key, exa_api_key)
    else:
        return get_gemini_response_basic(user_message, api_key)

def perform_analysis_with_progress(user_message: str, api_key: str, exa_api_key: Optional[str] = None) -> Dict:
    """Perform analysis with enhanced visual progress indicators and spinners"""
    import time
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔬 Analisis Hybrid AI Sedang Berlangsung...")
        
        # Initialize progress components
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_details = st.empty()
        
        try:
            # Check if hybrid mode is enabled
            is_hybrid = HYBRID_AVAILABLE and st.session_state.get('use_hybrid_mode', True)
            has_exa = exa_api_key is not None
            
            # Step 1: Initialize Analysis (15%)
            with st.spinner("Memulai analisis..."):
                status_text.markdown("**🧠 Langkah 1/6:** Memulai analisis dengan Gemini AI...")
                step_details.info("⏳ Memproses gejala dan mempersiapkan analisis...")
                progress_bar.progress(15)
                time.sleep(0.4)
            
            # Step 2: Initial Processing (25%)
            with st.spinner("Menganalisis gejala..."):
                status_text.markdown("**📋 Langkah 2/6:** Menganalisis gejala yang dilaporkan...")
                step_details.info("🔍 Mengekstrak dan mengkategorikan gejala...")
                progress_bar.progress(25)
                time.sleep(0.3)
            
            if is_hybrid:
                # Step 3: Preparing Hybrid Systems (40%)
                with st.spinner("Mempersiapkan sistem hybrid..."):
                    status_text.markdown("**⚙️ Langkah 3/6:** Mempersiapkan sistem hybrid...")
                    step_details.info("⚙️ Menginisialisasi RAG dan sistem pencarian...")
                    progress_bar.progress(40)
                    time.sleep(0.5)
                
                # Step 4: Document & Web Search (60%)
                with st.spinner("Mencari informasi medis..."):
                    if has_exa:
                        status_text.markdown("**📚🌐 Langkah 4/6:** Mencari pedoman & penelitian...")
                        step_details.info("🌐 Mengakses database penelitian medis dan pedoman lokal...")
                    else:
                        status_text.markdown("**📚 Langkah 4/6:** Mencari pedoman medis...")
                        step_details.info("� Mengakses database pedoman medis lokal...")
                    progress_bar.progress(60)
                    time.sleep(0.6)
                
                # Step 5: Processing Information (80%)
                with st.spinner("Memproses informasi medis..."):
                    status_text.markdown("**🧠 Langkah 5/6:** Memproses informasi medis...")
                    step_details.info("🧠 Sistem AI sedang menganalisis data dari berbagai sumber...")
                    progress_bar.progress(80)
                    time.sleep(0.3)
                
            else:
                # Basic mode steps
                with st.spinner("Menggunakan analisis standar..."):
                    status_text.markdown("**📖 Langkah 3/6:** Mode analisis standar...")
                    step_details.info("� Menggunakan analisis dasar...")
                    progress_bar.progress(40)
                    time.sleep(0.4)
                
                with st.spinner("Memproses dengan AI..."):
                    status_text.markdown("**🔬 Langkah 4/6:** Memproses dengan AI...")
                    step_details.info("� Menganalisis gejala dengan Gemini AI...")
                    progress_bar.progress(60)
                    time.sleep(0.5)
                
                with st.spinner("Menyusun hasil analisis..."):
                    status_text.markdown("**⚕️ Langkah 5/6:** Menyusun hasil...")
                    step_details.info("📝 AI sedang menyusun diagnosis dan rekomendasi...")
                    progress_bar.progress(80)
                    time.sleep(0.3)
            
            # Final processing with spinner for the actual AI call
            with st.spinner("Menyelesaikan analisis medis..."):
                step_details.info("⚕️ Menyelesaikan diagnosis dan rekomendasi...")
                time.sleep(0.2)
                
                # Perform the actual analysis
                result = get_gemini_response(user_message, api_key, exa_api_key)
            
            # Step 6: Complete (100%) with final spinner
            with st.spinner("Menyelesaikan dan memformat hasil..."):
                status_text.markdown("**✅ Langkah 6/6:** Analisis selesai!")
                if result.get('sources_used'):
                    sources = result['sources_used']
                    step_details.success(f"🎯 Berhasil menggunakan {sources.get('total_sources', 0)} sumber: {sources.get('local_guidelines', 0)} pedoman lokal + {sources.get('web_research', 0)} penelitian web")
                else:
                    step_details.success("🎯 Analisis dasar selesai dengan hasil yang akurat!")
                progress_bar.progress(100)
                
                # Brief pause to show completion
                time.sleep(0.8)
            
            return result
            
        except Exception as e:
            status_text.error("❌ Terjadi kesalahan dalam analisis")
            step_details.error(f"Error: {str(e)}")
            # Return basic analysis as fallback
            return get_gemini_response_basic(user_message, api_key)
        
        finally:
            # Clean up progress bar after a brief delay
            time.sleep(0.5)
            progress_container.empty()

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
        range_x=[0, 100]
    )
    
    fig.update_layout(
        height=max(300, len(conditions) * 60),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Add percentage labels on bars
    fig.update_traces(
        texttemplate='%{x}%',
        textposition='outside'
    )
    
    return fig

def get_risk_badge_color(risk_level):
    """Get color for risk level badge with medical triage colors"""
    colors = {
        'low': '#28a745',      # Green - Non-urgent (Prioritas III)
        'medium': '#ffc107',   # Yellow - Urgent (Prioritas II) 
        'high': '#dc3545',     # Red - Critical (Prioritas I)
        'emergency': '#6f42c1' # Purple - Immediate (Prioritas 0)
    }
    return colors.get(risk_level.lower(), '#6c757d')

def display_triage_results(triage_data):
    """Display triage results with enhanced formatting and medical context"""
    if not triage_data:
        return
    
    urgency = triage_data.get('urgency', 'medium')
    priority = triage_data.get('priority', 3)
    recommendation = triage_data.get('recommendation', 'Konsultasi dengan dokter')
    reasoning = triage_data.get('reasoning', 'Berdasarkan analisis gejala')
    
    # Map urgency to medical triage categories and ensure consistency
    urgency_mapping = {
        'low': ('🟢 NON-URGEN', 'Prioritas III - Ringan', 1),
        'medium': ('🟡 URGEN', 'Prioritas II - Serius tapi stabil', 3),
        'high': ('🔴 KRITIS', 'Prioritas I - Mengancam nyawa', 4),
        'emergency': ('🔴 IMMEDIATE', 'Prioritas 0 - Segera ditangani', 5)
    }
    
    urgency_display, priority_description, expected_priority = urgency_mapping.get(urgency.lower(), ('🟡 URGEN', 'Prioritas II', 3))
    
    # Use consistent priority - if the priority doesn't match urgency, use the mapped one
    if abs(priority - expected_priority) > 1:  # If there's a mismatch
        consistent_priority = expected_priority
    else:
        consistent_priority = priority
    
    # Create columns for triage display
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown("### 🚨 Tingkat Urgensi")
        color = get_risk_badge_color(urgency)
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
            {urgency_display}
        </div>
        <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">
            {priority_description}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Skala Prioritas")
        # Use color based on consistent priority level
        if consistent_priority >= 4:
            priority_color = "#dc3545"  # Red for high priority
        elif consistent_priority >= 3:
            priority_color = "#ffc107"  # Yellow for medium priority
        else:
            priority_color = "#28a745"  # Green for low priority
            
        st.markdown(f"""
        <div style="background-color: {priority_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 24px;">
            {consistent_priority}/5
        </div>
        <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">
            Skala 1-5 (5 = paling mendesak)
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 💡 Rekomendasi")
        st.info(recommendation)
    
    # Reasoning section
    st.markdown("### 🔍 Alasan Medis")
    st.write(reasoning)
    
    # Emergency contact info for high priority cases
    if urgency in ['high', 'emergency'] or priority >= 4:
        st.error("""
        ⚠️ **PERHATIAN**: Kondisi ini memerlukan perhatian medis segera!
        
        📞 **Kontak Darurat:**
        - Rumah Sakit Terdekat: 119
        - Ambulans: 118
        - Puskesmas: (021) xxx-xxxx
        """)
    
    return triage_data

def display_full_analysis_results(analysis_result):
    """Display complete analysis results including conditions, recommendations, etc."""
    if not analysis_result:
        return
    
    # Show extracted symptoms if available
    if st.session_state.get("collected_symptoms"):
        st.markdown("### 📋 Gejala yang Terkumpul")
        all_symptoms = []
        for symptom_text in st.session_state.collected_symptoms:
            all_symptoms.extend(extract_symptoms_simple(symptom_text))
        
        if all_symptoms:
            unique_symptoms = list(dict.fromkeys(all_symptoms))  # Remove duplicates
            cols = st.columns(min(3, len(unique_symptoms)))
            for i, symptom in enumerate(unique_symptoms):
                with cols[i % 3]:
                    st.success(f"✓ {symptom}")
        st.markdown("---")
    
    # Display triage first
    if 'triage' in analysis_result:
        display_triage_results(analysis_result['triage'])
    
    st.markdown("---")
    
    # Conditions analysis
    if 'conditions' in analysis_result and analysis_result['conditions']:
        st.markdown("### 🔍 Possible Conditions")
        
        # Create likelihood chart
        chart = create_likelihood_chart(analysis_result['conditions'])
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Display detailed conditions
        for i, condition in enumerate(analysis_result['conditions']):
            with st.expander(f"📋 {condition.get('name', 'Unknown Condition')} ({condition.get('likelihood', 0)}% likelihood)"):
                st.write(f"**Description:** {condition.get('description', 'No description available')}")
                
                if condition.get('symptoms'):
                    st.write("**Associated Symptoms:**")
                    for symptom in condition['symptoms']:
                        st.write(f"• {symptom}")
    
    # Recommendations
    if 'recommendations' in analysis_result and analysis_result['recommendations']:
        st.markdown("### 💡 Recommendations")
        for rec in analysis_result['recommendations']:
            st.success(f"✓ {rec}")
    
    # Red flags
    if 'red_flags' in analysis_result and analysis_result['red_flags']:
        st.markdown("### 🚨 Warning Signs")
        for flag in analysis_result['red_flags']:
            st.error(f"⚠️ {flag}")
    
    # Follow-up
    if 'follow_up' in analysis_result and analysis_result['follow_up']:
        st.markdown("### 📅 Follow-up")
        st.info(analysis_result['follow_up'])
    
    # Sources information (for hybrid mode)
    if analysis_result.get('sources_used'):
        sources = analysis_result['sources_used']
        with st.expander("📚 Information Sources"):
            st.write(f"**Total Sources Used:** {sources.get('total_sources', 0)}")
            st.write(f"• Local Medical Guidelines: {sources.get('local_guidelines', 0)}")
            st.write(f"• Web Research Papers: {sources.get('web_research', 0)}")
            
            if sources.get('sources'):
                st.write("**Source Details:**")
                for source in sources['sources'][:5]:  # Show first 5 sources
                    st.write(f"• {source.get('title', 'Medical Source')}")

def get_related_symptoms(condition_name):
    """Get related symptoms for a condition"""
    symptom_database = {
        "flu": ["demam", "batuk", "pilek", "sakit kepala", "nyeri otot", "kelelahan"],
        "covid-19": ["demam", "batuk kering", "sesak napas", "hilang penciuman", "hilang pengecapan", "kelelahan"],
        "dengue": ["demam tinggi", "sakit kepala", "nyeri mata", "nyeri otot", "ruam kulit", "mual"],
        "typhoid": ["demam", "sakit kepala", "nyeri perut", "diare", "konstipasi", "ruam"],
        "gastritis": ["nyeri perut", "mual", "muntah", "kembung", "heartburn"],
        "hipertensi": ["sakit kepala", "pusing", "sesak napas", "nyeri dada", "penglihatan kabur"],
        "diabetes": ["sering haus", "sering buang air kecil", "kelelahan", "penglihatan kabur", "luka sulit sembuh"],
        "asma": ["sesak napas", "mengi", "batuk", "dada sesak"],
        "migrain": ["sakit kepala berdenyut", "mual", "muntah", "sensitif cahaya", "sensitif suara"],
        "pneumonia": ["batuk", "demam", "sesak napas", "nyeri dada", "kelelahan"]
    }
    
    related = []
    condition_lower = condition_name.lower()
    
    for condition, symptoms in symptom_database.items():
        if condition in condition_lower or any(word in condition_lower for word in condition.split()):
            related.extend(symptoms)
    
    # Remove duplicates and return
    return list(dict.fromkeys(related))

def extract_symptoms_simple(user_input):
    """Extract symptoms from user input using simple keyword matching"""
    
    # Common symptoms in Indonesian
    symptom_keywords = {
        "demam": ["demam", "panas", "fever", "hot"],
        "batuk": ["batuk", "cough"],
        "pilek": ["pilek", "ingus", "runny nose", "hidung tersumbat"],
        "sakit kepala": ["sakit kepala", "pusing", "headache", "dizzy"],
        "mual": ["mual", "nausea", "pengen muntah"],
        "muntah": ["muntah", "vomit"],
        "diare": ["diare", "mencret", "diarrhea", "BAB cair"],
        "konstipasi": ["konstipasi", "sembelit", "susah BAB", "constipation"],
        "nyeri perut": ["sakit perut", "nyeri perut", "perut sakit", "stomach pain"],
        "sesak napas": ["sesak napas", "susah bernapas", "shortness of breath"],
        "nyeri dada": ["sakit dada", "nyeri dada", "chest pain"],
        "kelelahan": ["lelah", "capek", "fatigue", "tired"],
        "nyeri otot": ["nyeri otot", "pegal", "muscle pain"],
        "ruam": ["ruam", "bintik merah", "rash"],
        "gatal": ["gatal", "itchy"],
        "bengkak": ["bengkak", "swelling"],
        "berkeringat": ["berkeringat", "sweating", "keringat berlebih"]
    }
    
    user_input_lower = user_input.lower()
    extracted = []
    
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                extracted.append(symptom)
                break
    
    return list(dict.fromkeys(extracted))

def get_related_symptoms_simple(extracted_symptoms):
    """Get related symptoms based on extracted symptoms"""
    
    # Symptom associations
    associations = {
        "demam": ["sakit kepala", "kelelahan", "nyeri otot", "berkeringat"],
        "batuk": ["pilek", "sakit kepala", "kelelahan"],
        "pilek": ["batuk", "sakit kepala"],
        "sakit kepala": ["demam", "mual", "kelelahan"],
        "mual": ["muntah", "sakit kepala", "nyeri perut"],
        "muntah": ["mual", "nyeri perut", "kelelahan"],
        "diare": ["nyeri perut", "mual", "kelelahan"],
        "nyeri perut": ["mual", "muntah", "diare"],
        "sesak napas": ["batuk", "nyeri dada", "kelelahan"],
        "nyeri dada": ["sesak napas", "kelelahan"],
        "kelelahan": ["demam", "sakit kepala", "nyeri otot"],
        "ruam": ["gatal", "demam"],
        "gatal": ["ruam"]
    }
    
    related = []
    for symptom in extracted_symptoms:
        if symptom in associations:
            related.extend(associations[symptom])
    
    # Remove duplicates and original symptoms
    related = [s for s in list(dict.fromkeys(related)) if s not in extracted_symptoms]
    
    return related

def get_related_symptoms_from_extraction(extracted_symptoms, condition_name):
    """Combine extracted symptoms with condition-based related symptoms"""
    
    # Get symptoms from extraction
    related_from_symptoms = get_related_symptoms_simple(extracted_symptoms)
    
    # Get symptoms from condition
    related_from_condition = get_related_symptoms(condition_name)
    
    # Combine and deduplicate
    all_related = related_from_symptoms + related_from_condition
    
    # Remove duplicates and already mentioned symptoms
    unique_related = []
    for symptom in all_related:
        if symptom not in unique_related and symptom not in extracted_symptoms:
            unique_related.append(symptom)
    
    return unique_related[:8]  # Limit to 8 related symptoms

def load_guidelines():
    """Load medical guidelines from files"""
    guidelines = {}
    guidelines_dir = "guidelines"
    
    if os.path.exists(guidelines_dir):
        for filename in os.listdir(guidelines_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(guidelines_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        # Use filename without extension as key
                        key = filename[:-3]
                        guidelines[key] = content
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
    
    docs = [{"content": content, "source": key} for key, content in guidelines.items()]
    return docs


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Patient Symptom Analysis Chatbot</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This chatbot is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings and information
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        def clear_conversation():
            keys_to_clear = [
                'chat_history', 'symptom_collection_mode', 'collected_symptoms',
                'selected_additional_symptoms', 'trigger_analysis', 'prev_question_timestamp'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
        
        # Hybrid Mode Toggle
        if HYBRID_AVAILABLE:
            st.markdown("### 🔬 AI Analysis Mode")
            use_hybrid = st.toggle(
                "Enable Hybrid Mode", 
                value=st.session_state.get('use_hybrid_mode', True),
                help="Uses local medical guidelines + web research for enhanced analysis"
            )
            st.session_state['use_hybrid_mode'] = use_hybrid
            
            if use_hybrid:
                st.success("🚀 Hybrid Mode: Enhanced AI with medical research")
            else:
                st.info("📖 Basic Mode: Standard AI analysis")
        else:
            st.warning("📖 Basic Mode Only\n\nInstall hybrid dependencies for enhanced features")
        
        # Medical Priority Guide
        st.markdown("### 🏥 Panduan Prioritas Medis")
        if st.button("ℹ️ Penjelasan Tingkat Prioritas Triase", use_container_width=True):
            # Close any other modals first
            st.session_state.show_tech_modal = False
            st.session_state.show_triage_modal = True
            st.rerun()
        
        # Technology Stack
        st.markdown("### 🔧 Technology Stack")
        if st.button("⚡ Technologies Used in This Chatbot", use_container_width=True):
            # Close any other modals first
            st.session_state.show_triage_modal = False
            st.session_state.show_tech_modal = True
            st.rerun()
        
        # Developer Information
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Created by:** [Enzi Muzakki](https://www.linkedin.com/in/enzimuzakki/)  
        **LinkedIn:** [linkedin.com/in/enzimuzakki](https://www.linkedin.com/in/enzimuzakki/)  
        **GitHub:** [patient-symptom-chatbot](https://github.com/enzeeeh/patient-symptom-chatbot)
        """)
        
        # Debug section for developers
        st.markdown("### 🛠️ Developer Options")
        debug_mode = st.checkbox("Enable Debug Mode", key="debug_mode", help="Show technical errors and debug information")
        
        if debug_mode:
            debug_models = st.checkbox("🔍 Show Available Models", key="debug_models")
            if debug_models:
                st.info("Click 'Check Available Models' button below to see which models work with your API key.")
    
    # API Key Management
    def get_api_key():
        # Try to get from Streamlit secrets first
        try:
            if "api_keys" in st.secrets:
                return st.secrets["api_keys"]["gemini_api_key"]
            elif "gemini_api_key" in st.secrets:
                return st.secrets["gemini_api_key"]
        except:
            pass
        
        # Try environment variable
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            return env_key
        
        # Finally check session state
        if 'user_api_key' in st.session_state and st.session_state.user_api_key:
            return st.session_state.user_api_key
            
        return None

    api_key = get_api_key()
    
    if not api_key:
        st.warning("🔑 Gemini API key not found")
        with st.expander("Configure API Key"):
            user_api_key = st.text_input(
                "Enter your Gemini API Key:",
                type="password",
                help="Get your API key from https://makersuite.google.com/app/apikey"
            )
            if user_api_key:
                st.session_state.user_api_key = user_api_key
                st.success("✅ API key configured! You can now use the chatbot.")
                st.rerun()
            
            st.info("""
            **How to get a Gemini API key:**
            1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API key"
            4. Copy and paste the key above
            """)
        return
    
    # Exa API Key for enhanced web search
    def get_exa_api_key():
        try:
            if "exa_api_key" in st.secrets:
                return st.secrets["exa_api_key"]
        except:
            pass
        
        env_key = os.getenv('EXA_API_KEY')
        if env_key:
            return env_key
            
        if 'user_exa_key' in st.session_state and st.session_state.user_exa_key:
            return st.session_state.user_exa_key
            
        return None

    exa_api_key = get_exa_api_key()
    
    # Debug: Check available models if requested
    if st.session_state.get('debug_models', False):
        with st.sidebar:
            if st.button("🔍 Check Available Models"):
                with st.spinner("Checking available models..."):
                    models = list_available_models(api_key)
                    st.write("Available models:")
                    for model in models[:10]:  # Show first 10
                        st.write(f"• {model}")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Main interface header
    st.markdown("## 💬 Jelaskan Gejala Anda")
    
    # Example symptoms buttons
    st.markdown("### 📋 Contoh Gejala Umum (Klik untuk mulai)")
    
    example_symptoms = [
        "Saya mengalami demam tinggi dan sakit kepala sejak 2 hari",
        "Batuk kering, sesak napas, dan kelelahan",
        "Nyeri perut, mual, dan diare",
        "Pusing, nyeri dada, dan jantung berdebar",
        "Sakit tenggorokan dan demam ringan"
    ]
    
    cols = st.columns(2)
    for i, symptom in enumerate(example_symptoms):
        with cols[i % 2]:
            if st.button(f"🔸 {symptom}", key=f"example_{i}", use_container_width=True):
                st.session_state.selected_symptom = symptom

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
            # Display full analysis results if present
            if chat.get("triage"):
                display_full_analysis_results(chat["triage"])

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
        
        # Combine chat input with any selected symptoms from multiselect
        combined_symptoms = []
        
        # Add previously selected symptoms from multiselect
        if st.session_state.get("selected_additional_symptoms", []):
            selected_symptoms_text = f"Saya juga mengalami: {', '.join(st.session_state.selected_additional_symptoms)}"
            combined_symptoms.append(selected_symptoms_text)
            # Clear the selected symptoms since we're adding them
            st.session_state.selected_additional_symptoms = []
        
        # Add the new chat input
        combined_symptoms.append(user_input)
        
        # Add combined symptoms to collection
        for symptom in combined_symptoms:
            st.session_state.collected_symptoms.append(symptom)
        
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
        st.markdown("## 🔍 Apakah Anda mengalami gejala tambahan berikut?")
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
                tags_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 12px; display: inline-block;">🔸 {symptom}</span>'
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
                st.rerun()

    # Handle analysis trigger (must be outside the user_input block)
    if st.session_state.get("trigger_analysis", False):
        # Clear the trigger flag first
        st.session_state.trigger_analysis = False
        
        # Debug: Show what symptoms we have
        collected = st.session_state.get("collected_symptoms", [])
        
        # Perform final analysis with all collected symptoms
        if collected:
            all_symptoms = " dan ".join(collected)
            final_input = f"Ringkasan gejala lengkap: {all_symptoms}"
            
            # Use the enhanced progress bar for analysis
            final_triage = perform_analysis_with_progress(final_input, api_key, exa_api_key)
            
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

def get_relevant_guideline(conditions, symptoms):
    """Get relevant medical guideline based on conditions and symptoms"""
    docs = load_guidelines()
    if not docs:
        return "No guidelines available. Please consult with a healthcare professional for proper medical advice."
    
    # Simple keyword matching to find relevant guidelines
    relevant_content = []
    
    for doc in docs:
        content = doc['content'].lower()
        source = doc['source']
        
        # Check if any condition or symptom matches the guideline content
        for condition in conditions:
            condition_name = condition.get('name', '').lower()
            if any(word in content for word in condition_name.split() if len(word) > 3):
                relevant_content.append(f"From {source}:\n{doc['content']}")
                break
    
    if relevant_content:
        # Return first relevant guideline (truncated)
        content = relevant_content[0]
        return content[:1000] + "..." if len(content) > 1000 else content
    else:
        # Fallback to general guideline
        fallback = docs[0]['content'] if docs else "Please consult with a healthcare professional."
        return fallback[:1000] + "..." if len(fallback) > 1000 else fallback

@st.dialog("🏥 Panduan Prioritas Medis - Sistem Triase", width="large")
def show_triage_modal():
    # Add custom CSS for wider modal
    st.markdown("""
    <style>
    div[data-testid="modal"] > div {
        width: 90% !important;
        max-width: 1000px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Sistem Triase Medis** digunakan untuk menentukan urgensi penanganan pasien:
    
    | Warna | Makna | Triase | Skala 1-5 | Kondisi Contoh |
    |-------|--------|---------|-----------|----------------|
    | 🔴 **Merah** | **Immediate/Kritis** | **Prioritas 0-I** | **5/5** | Mengancam nyawa segera (henti napas, syok, pendarahan masif). |
    | 🔴 **Merah** | **Kritis** | **Prioritas I** | **4/5** | Mengancam nyawa tapi stabil beberapa menit (stroke akut, serangan jantung). |
    | 🟡 **Kuning** | **Urgen** | **Prioritas II** | **3/5** | Serius tapi stabil, dapat ditunda beberapa jam (patah tulang besar, luka bakar sedang). |
    | 🟢 **Hijau** | **Non-Urgen** | **Prioritas III** | **1-2/5** | Ringan, tidak mengancam nyawa (luka lecet, demam ringan). |
    | ⚫ **Hitam** | **Meninggal** | **Prioritas 0** | **-** | Sudah meninggal atau tidak dapat diselamatkan. |
    
    **Penjelasan Skala:**
    - **5/5**: Segera (dalam hitungan menit)
    - **4/5**: Sangat mendesak (dalam 1 jam)
    - **3/5**: Mendesak (dalam beberapa jam)
    - **1-2/5**: Tidak mendesak (dapat menunggu)
    
    **Catatan:** 
    - Tingkat urgensi ini hanya sebagai panduan awal
    - Selalu konsultasikan dengan tenaga medis profesional
    - Jika ragu, lebih baik segera mencari bantuan medis
    
    **Sumber:**
    - Pedoman WHO untuk Sistem Triase
    - Standar International Emergency Nursing
    """)
    
    if st.button("Tutup", type="primary", use_container_width=True):
        # Clear all modal flags
        st.session_state.show_triage_modal = False
        st.session_state.show_tech_modal = False
        st.rerun()

@st.dialog("🔧 Technology Stack - Patient Symptom Chatbot", width="large")
def show_tech_modal():
    # Add custom CSS for wider modal
    st.markdown("""
    <style>
    div[data-testid="modal"] > div {
        width: 90% !important;
        max-width: 1000px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if HYBRID_AVAILABLE:
        st.markdown("""
        **🚀 Hybrid AI System:**
        - **Core AI:** Google Gemini (gemini-1.5-flash-001 / gemini-pro-001)
        - **RAG System:** LangChain for local medical guidelines processing
        - **Embeddings:** Google Generative AI Embeddings + HuggingFace fallback
        - **Vector Store:** FAISS for fast similarity search
        - **Web Research:** Exa API for real-time medical research
        - **Document Processing:** RecursiveCharacterTextSplitter
        
        **📚 Data Sources:**
        - Local medical guidelines (COVID-19, Dengue, Typhoid, etc.)
        - Real-time web research for latest medical information
        - WHO and medical authority guidelines
        
        **🛠️ Framework & Infrastructure:**
        - **Frontend:** Streamlit with custom CSS styling
        - **Backend:** Python with async processing
        - **Deployment:** Streamlit Cloud
        - **Version Control:** Git & GitHub
        """)
    else:
        st.markdown("""
        **📖 Basic AI System:**
        - **Core AI:** Google Gemini (gemini-1.5-flash-001 / gemini-pro-001)
        - **Data Sources:** Built-in medical knowledge base
        - **Framework:** Streamlit with custom CSS styling
        - **Deployment:** Streamlit Cloud
        - **Version Control:** Git & GitHub
        
        **💡 Upgrade to Hybrid Mode for:**
        - Local medical guideline processing
        - Real-time web research capabilities
        - Enhanced accuracy with RAG system
        """)
    
    st.markdown("""
    **🎨 User Experience:**
    - Progress bars with animated spinners
    - Color-coded medical priority system
    - Responsive design with medical theme
    - Professional medical triage guidance
    
    **🔒 Security & Privacy:**
    - API key encryption in Streamlit secrets
    - No personal data storage
    - Secure HTTPS communication
    
    **⚡ Performance:**
    - Optimized AI model selection with fallbacks
    - Efficient vector search for medical guidelines
    - Real-time web research integration
    - Smart caching for improved response times
    """)
    
    if st.button("Tutup", type="primary", use_container_width=True):
        # Clear all modal flags
        st.session_state.show_triage_modal = False
        st.session_state.show_tech_modal = False
        st.rerun()

if __name__ == "__main__":
    # Show modals if requested - only one at a time
    if st.session_state.get('show_triage_modal', False):
        show_triage_modal()
    elif st.session_state.get('show_tech_modal', False):
        show_tech_modal()
    
    main()