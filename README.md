# Gig Worker Law Assistant

A multilingual chatbot that answers questions about the gig-worker laws of Rajasthan, Karnataka,
Telangana and Jharkhand (English, Hindi, Hinglish, Tamil). Each answer shows the three law passages it
was based on. Information only, not legal advice.

Built with Streamlit, LangChain, a multilingual MiniLM sentence encoder, FAISS and gpt-oss-20b on Groq.

## Run on Streamlit Community Cloud
1. Put `app.py`, `requirements.txt` and the `policies/` folder in a public GitHub repository.
2. At https://share.streamlit.io, create an app from that repository with main file `app.py`.
3. In Advanced settings, choose Python 3.11 and add the secret:
   `GROQ_API_KEY = "your-groq-key"`

## Run on your own computer
```
pip install -r requirements.txt
set GROQ_API_KEY=your-groq-key        (Windows)   |   export GROQ_API_KEY=your-groq-key   (Mac/Linux)
streamlit run app.py
```
To add another state's law, put its PDF in `policies/` (the file name becomes the source label) and restart.
