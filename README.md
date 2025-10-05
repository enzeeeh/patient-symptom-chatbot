# 🏥 Patient Symptom Analysis Chatbot

An advanced AI-powered medical triage chatbot built with Streamlit and Google Gemini AI that provides comprehensive symptom analysis and medical guidance. Features hybrid AI architecture with local medical guidelines and real-time web research capabilities.

## 🌟 Key Features

### 🤖 **Hybrid AI System**
- **Dual-Mode Analysis**: Basic mode with Gemini AI + Enhanced hybrid mode with RAG system
- **Local Medical Guidelines**: WHO, CDC, and medical authority guidelines integration
- **Real-time Web Research**: Exa API integration for latest medical information
- **Smart Model Fallback**: Automatic fallback between gemini-1.5-flash-001 and gemini-pro-001

### 🎯 **Advanced User Experience**
- **Interactive Symptom Collection**: Multi-step guided symptom selection with smart suggestions
- **Progress Tracking**: 6-step visual progress bars with animated spinners
- **Wide Modal Dialogs**: Professional pop-up system for medical triage guide and tech stack
- **Enhanced Visual Analytics**: Likelihood charts with condition analysis
- **Professional UI**: Color-coded medical priority system with triage badges

### 📊 **Medical Intelligence**
- **Medical Triage System**: WHO-standard 5-point priority scale (1-5) with color coding
- **Symptom Extraction**: Advanced NLP for automatic symptom detection
- **Risk Stratification**: Emergency, high, medium, and low priority classification
- **Smart Recommendations**: Personalized medical advice with red flag warnings

## 🏥 Medical Capabilities

### Symptom Recognition
- Fever and chills
- Headache and migraines
- Respiratory symptoms (cough, shortness of breath)
- Gastrointestinal issues (nausea, vomiting, diarrhea)
- Pain assessment (chest, abdominal, muscle)
- General symptoms (fatigue, loss of appetite)

### Condition Analysis
- Infectious diseases (dengue, influenza, typhoid)
- Respiratory conditions
- Gastrointestinal disorders
- General medical conditions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Exa API key (optional, for enhanced web research)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/enzeeeh/patient-symptom-chatbot.git
cd patient-symptom-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys (choose one method):

**Option A: Streamlit Secrets (Recommended)**
Create `.streamlit/secrets.toml`:
```toml
gemini_api_key = "your-gemini-api-key"
exa_api_key = "your-exa-api-key"  # Optional
```

**Option B: Environment Variables**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export EXA_API_KEY="your-exa-api-key"  # Optional
```

**Option C: In-App Configuration**
- API keys can be entered directly in the app interface

4. Run the application:
```bash
streamlit run app.py
```

## 🔧 Technology Stack

### **🚀 Hybrid AI System**
- **Core AI**: Google Gemini (gemini-1.5-flash-001 / gemini-pro-001)
- **RAG System**: LangChain for local medical guidelines processing
- **Embeddings**: Google Generative AI Embeddings + HuggingFace fallback
- **Vector Store**: FAISS for fast similarity search
- **Web Research**: Exa API for real-time medical research
- **Document Processing**: RecursiveCharacterTextSplitter

### **🛠️ Framework & Infrastructure**
- **Frontend**: Streamlit with custom CSS styling and modal dialogs
- **Backend**: Python with async processing
- **Visualization**: Plotly Express for interactive charts
- **Data Processing**: Pandas for data manipulation
- **Deployment**: Streamlit Cloud ready

## 📋 How to Use

### **Basic Workflow**
1. **🎯 Choose Analysis Mode**: Select Basic Mode or Enhanced Hybrid Mode
2. **💬 Describe Symptoms**: Type symptoms or choose from common examples
3. **✅ Interactive Selection**: Select additional related symptoms from smart suggestions
4. **🔬 AI Analysis**: Watch the 6-step progress bar with detailed status updates
5. **📊 Review Results**: Get comprehensive triage analysis with visual charts
6. **📋 Follow Recommendations**: View personalized medical advice and next steps

### **🔍 Advanced Features**
- **📚 Medical Triage Guide**: Click modal button for WHO-standard priority explanations
- **⚡ Technology Information**: View complete tech stack in wide modal dialog
- **🎨 Visual Analytics**: Interactive likelihood charts and condition analysis
- **⚠️ Red Flag Warnings**: Automatic detection of emergency symptoms
- **📱 Responsive Design**: Works perfectly on desktop and mobile devices

## 🌍 Language & Localization

- **🇮🇩 Bahasa Indonesia**: Complete interface with medical terminology
- **🌐 English**: Comprehensive English support for international users
- **🔤 Smart Detection**: Automatic symptom recognition in both languages
- **📖 Medical Guidelines**: Localized medical advice and recommendations

## ⚠️ Medical Disclaimer

This application is for educational and preliminary assessment purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical care.

## 🏥 Medical Triage System

### **Priority Classification**
- **🔴 Priority I (4-5/5)**: Critical/Emergency - Immediate medical attention required
- **🟡 Priority II (3/5)**: Urgent - Medical evaluation within 24-48 hours
- **🟢 Priority III (1-2/5)**: Non-urgent - Can wait for routine medical care

### **Built-in Medical Guidelines**
- **WHO Dengue Clinical Management Guidelines**
- **WHO Diarrheal Disease Treatment Guidelines**
- **CDC Influenza Management Protocols**
- **WHO Typhoid Fever Guidelines**
- **Heart.org Chest Pain Assessment Guidelines**
- **COVID-19 Clinical Management Protocols**

## 🎨 UI/UX Features

### **Professional Modal System**
- **📋 Medical Triage Guide**: Wide modal with comprehensive WHO triage explanations
- **⚡ Technology Stack**: Detailed modal showing complete system architecture
- **🔧 Mutual Exclusion**: Only one modal open at a time for clean UX
- **📱 Responsive Design**: 90% width modals with 1000px maximum width

### **Enhanced Progress Tracking**
- **6-Step Visual Progress**: Real-time analysis status with detailed descriptions
- **🌀 Animated Spinners**: Visual feedback during each processing stage
- **📊 Status Updates**: Clear communication of current analysis step
- **⚡ Performance Indicators**: Source count and processing metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## �‍💻 Developer

**Created by:** [Enzi Muzakki](https://www.linkedin.com/in/enzimuzakki/)  
**LinkedIn:** [linkedin.com/in/enzimuzakki](https://www.linkedin.com/in/enzimuzakki/)  
**GitHub:** [enzeeeh/patient-symptom-chatbot](https://github.com/enzeeeh/patient-symptom-chatbot)

## 🙏 Acknowledgments

- **Google Gemini AI** for advanced language processing and medical analysis capabilities
- **Streamlit** for the incredible web framework and modal dialog system
- **LangChain** for RAG system implementation and document processing
- **Exa API** for real-time web research and medical information retrieval
- **WHO & CDC** for comprehensive medical guidelines and triage standards
- **Medical professionals** who inspired this project and provided guidance
- **Open source community** for amazing tools and libraries