# Multi-Agent & Robotic System (MARS) for Material Science Research

This project is a comprehensive platform for material science research, consisting of a backend API server, a frontend web application, and a middleware desktop application. The platform integrates various AI agents for planning, scientific analysis, engineering tasks, and data analysis to assist in material science research.

## Project Structure

The project is divided into three main components:

1. Backend (Python/FastAPI/Autogen)
2. Frontend (Vue.js)
3. Middleware (C#/WPF)

## Backend

The backend is built with Python using the FastAPI framework. It manages the core logic of the multi-agent system, handles WebSocket connections for real-time communication, and provides API endpoints for the frontend.

### Setup and Running

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies (it's recommended to use a virtual environment):
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the required API keys and configuration settings

4. Run the backend server:
   ```
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

The backend will be available at `http://localhost:8000`.

## Frontend

The frontend is a Vue.js application that provides the user interface for interacting with the multi-agent system.

### Setup and Running

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```

4. For production build:
   ```
   npm run build
   ```

Before building for production, ensure that the following environment variables in the `.env` file are correctly set:
- `VITE_API_URL`: The backend API URL
- `VITE_API_URL_PREFIX`: The API prefix
- `VITE_WB_BASE_URL`: The WebSocket base URL for real-time communication

## Middleware

The middleware is a C# WPF desktop application that provides additional functionality and integration with local systems.

### Setup and Running

1. Open the solution file `middleware/zdhsys.sln` in Visual Studio.

2. Build the solution in Visual Studio.

3. Run the application from Visual Studio or navigate to the build output directory and run the executable.

## Usage

1. Start the backend server.
2. Launch the frontend application (either in development mode or by serving the production build).
3. If required, run the middleware desktop application.
4. Access the web interface through your browser and begin interacting with the multi-agent material science research platform.

## Features

- Real-time communication using WebSockets
- Multi-agent system for complex task planning and execution
- Integration of scientific, engineering, and data analysis teams
- Video streaming capabilities for remote monitoring
- Customizable UI for different aspects of material science research

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
