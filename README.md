
# aichat-stream-mcp-template

This is a template project to quickstart a chat app using the Stream API, OpenAI, and MCP tools. It includes a backend (Python) and a frontend (React + Vite).

## Project Structure

- `backend/` - Python backend (FastAPI or Flask recommended)
  - `main.py` - Main backend application
  - `requirements.txt` - Python dependencies
- `frontend/` - React frontend (Vite)
  - `App.jsx`, `index.jsx` - Main React components
  - `ChatTools.jsx` - Chat tools UI
  - `global.css`, `ChatTools.module.css` - Styles
  - `index.html` - HTML entry point
  - `package.json` - Frontend dependencies

## Getting Started

### Backend Setup
1. Navigate to the `backend` folder:
	```sh
	cd backend
	```
2. Install Python dependencies:
	```sh
	pip install -r requirements.txt
	```
3. Run the backend server (update if using FastAPI or Flask):
	```sh
	python main.py
	```

### Backend .env
```
AZURE_OPENAI_ENDPOINT = 
AZURE_OPENAI_API_KEY = 
AZURE_OPENAI_API_VERSION = 
```

### Frontend Setup
1. Navigate to the `frontend` folder:
	```sh
	cd frontend
	```
2. Install Node.js dependencies:
	```sh
	npm install
	```
3. Start the frontend development server:
	```sh
	npm run dev
	```

## Usage
1. Start the backend server.
2. Start the frontend server.
3. Open your browser to the URL provided by Vite (usually `http://localhost:5173`).
4. Interact with the chat app, which streams responses from the backend using OpenAI and MCP tools.

## License
See `LICENSE` for details.
