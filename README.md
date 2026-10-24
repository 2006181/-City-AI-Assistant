
# 🌍 City AI Assistant

An AI-powered city assistant built using Streamlit, Google Gemini, LangChain, OpenWeatherMap, and Tavily Search.

Ask questions, check current weather conditions, and discover recent city news through a simple conversational interface.

## ✨ Features

- 💬 Interactive Streamlit chat interface
- 🤖 Google Gemini-powered AI assistant
- 🌦️ Current weather and temperature information
- 📰 Recent news headlines, summaries, and article links
- 🔎 Tavily-powered news search
- 🧰 Selectable assistant modes for chat, weather, news, or all tools
- 🧹 Clean AI responses without displaying internal signature metadata
- 🗂️ Chat history maintained during the current Streamlit session
- 🔐 API keys managed through environment variables

## 🛠️ Tech Stack

- Python
- Streamlit
- Google Gemini
- LangChain
- OpenWeatherMap API
- Tavily Search API
- python-dotenv
- Requests

## 📁 Project Structure

```text
City-AI-Assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd City-AI-Assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
TAVILY_API_KEY=your_tavily_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Obtain API credentials from:

- Google AI Studio: https://aistudio.google.com/
- OpenWeatherMap: https://openweathermap.org/api
- Tavily: https://tavily.com/

Use valid credentials and a Gemini model currently supported by your account.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

## 💡 How to Use

1. Start the application.
2. Select a mode from the sidebar.
3. Enter a question in the chat input.
4. Ask about weather, city news, or general topics.
5. Use Clear chat to start a new conversation.

### Available Modes

- **Chat only:** General conversation without tools.
- **Weather assistant:** Enables the weather tool.
- **News assistant:** Enables the news search tool.
- **All tools:** Enables both weather and news tools.

## 🔑 Environment Variables

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Authenticates Google Gemini |
| `OPENWEATHER_API_KEY` | Retrieves weather data |
| `TAVILY_API_KEY` | Searches for recent news |
| `GEMINI_MODEL` | Selects the Gemini model |

## 🔒 Security

- Never commit `.env` or API credentials to GitHub.
- Keep API keys private.
- Configure secrets securely when deploying.
- Review API usage limits and pricing.

## 🚀 Deployment

The application can be deployed on Streamlit Community Cloud or another compatible hosting platform.

Add the required environment variables in the deployment platform's secrets settings instead of uploading `.env`.

## ⚠️ Notes

- Weather and news functionality requires valid API keys.
- Search results and weather data depend on external services.
- The displayed chat history lasts for the active Streamlit session.
- The selected assistant mode controls which tools are available to the agent.

## 📄 License

This project is available for learning and personal portfolio use. Add a license file if you plan to distribute it under specific terms.
