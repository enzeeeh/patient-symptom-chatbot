# 💬 Patient Symptom Triage Chatbot

An intelligent medical triage chatbot built with Streamlit and Google Gemini AI that helps patients assess their symptoms and get preliminary medical guidance in both Indonesian and English.

## 🌟 Features

- **Bilingual Support**: Full interface in Bahasa Indonesia with English support
- **Smart Symptom Detection**: Automatic extraction of symptoms from user input
- **Interactive Symptom Collection**: Guided symptom selection with related symptom suggestions
- **AI-Powered Analysis**: Medical triage analysis using Google Gemini 2.5-flash
- **Risk Assessment**: Categorizes symptoms as low, moderate, or high risk
- **Visual Analytics**: Charts and graphs for condition likelihood analysis
- **Medical Guidelines**: Built-in reference to WHO and CDC medical guidelines
- **Comprehensive Results**: Detailed recommendations and next steps

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

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PatientSymptomChatbot.git
cd PatientSymptomChatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update the API key in `app.py`:
```python
api_key = "your-gemini-api-key-here"
```

4. Run the application:
```bash
streamlit run app.py
```

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5-flash
- **Visualization**: Plotly
- **Data Processing**: Pandas
- **Language**: Python

## 📋 Usage

1. **Start Conversation**: Choose from quick examples or type your symptoms
2. **Symptom Collection**: The AI extracts symptoms and suggests related ones
3. **Interactive Selection**: Use checkboxes to select additional symptoms
4. **Analysis**: Get comprehensive medical triage results
5. **Next Steps**: Options to start new analysis, copy results, or view guidelines

## 🌍 Language Support

- **Primary**: Bahasa Indonesia
- **Secondary**: English
- **Symptom Recognition**: Supports both languages simultaneously

## ⚠️ Medical Disclaimer

This application is for educational and preliminary assessment purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical care.

## 📖 Medical Guidelines References

- WHO Dengue Guidelines
- CDC Influenza Guidelines  
- WHO Diarrheal Disease Guidelines
- WHO Typhoid Guidelines
- Heart.org Chest Pain Guidelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Google Gemini AI for powerful language processing
- Streamlit for the amazing web framework
- WHO and CDC for medical guidelines
- Medical professionals who inspired this project