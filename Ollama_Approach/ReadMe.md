# Chat with Multiple PDFs

This project allows you to interact with uploaded PDF documents through a chat interface. The application uses Streamlit for the frontend, integrates language models with LangChain, and employs FAISS for efficient similarity searches.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or higher
- Conda
- pip (Python package installer)

## Installation

### Step 1: Install Ollama 
<!-- Just write Run this on shell curl -fsSL https://ollama.com/install.sh | sh
 -->
Run this on shell 
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull the Llama3 Model
After installing Ollama, pull the Llama3 model:

```sh
ollama pull llama3
```

### Step 3: Install Python Dependencies
Install the required Python packages using the provided requirements.txt:

```sh
pip install -r requirements.txt
```

## Usage
### Step 4: Run the Streamlit Application
Run the Streamlit application with the following command:

```sh
streamlit run app.py
```

## Project Structure
- app.py: The main application file.
- requirements.txt: List of Python dependencies.
- htmlTemplates.py: HTML templates used in the Streamlit app.