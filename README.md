LLM Insurance Query Assistant (POC)

An internal AI-powered insurance query assistant built using Hugging Face LLMs and Streamlit.
This project demonstrates how Large Language Models (LLMs) can help insurance teams retrieve, summarize, and answer policy-related questions efficiently.

Live Demo

🌐 Deployed App:
https://llm-poc-lcyhhdvzrtddwfueephpdt.streamlit.app/

GitHub Repository
https://github.com/hueywen20/llm-poc.git

📦 Source Code:
llm-poc Repository

Features
🤖 Hugging Face LLM integration
💬 Natural language insurance query support
📄 Policy and claim information summarization
⚡ Streamlit-based interactive UI
🔒 Internal-use proof of concept (POC)
☁️ Cloud deployment with Streamlit Community Cloud
Tech Stack
Python
Streamlit
Hugging Face Transformers
Hugging Face Inference API
Pandas
LangChain (if applicable)
dotenv

Installation
1. Clone the repository
   
```git clone https://github.com/hueywen20/llm-poc.git```
```cd llm-poc```

3. Create a virtual environment

```python -m venv venv```

Activate the environment:

Windows

```venv\Scripts\activate```

macOS/Linux

```source venv/bin/activate```

3. Install dependencies

```pip install -r requirements.txt```

4. Configure environment variables

Create a .env file:

```HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key```

You can generate a Hugging Face token from:
Hugging Face Settings Tokens

Running the Application

```streamlit run app.py```

The application will be available at:

http://localhost:8501
