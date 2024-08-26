# React PDF Viewer Application

This project is a replication task created for an internship assessment. It is a web application built with React and Node.js that enables users to enter text, retrieve answers and context with its related PDF files from a server, and display them directly in the browser. The application utilizes a JSON file to provide questions, answers, and PDF URLs as its data source. It also features a basic Express server to manage CORS and serve PDF files.
## Features

- **Interactive Chat Interface**: Users can input questions and receive answers along with relevant PDF sources.
- **PDF Display**: Embedded PDF viewer for viewing documents directly within the application.
- **Context Display**: Clickable context boxes that display excerpts from the source documents.
- **Data-Driven**: Utilizes a JSON file to load questions, answers, and source URLs.
- **New Chat Functionality**: Reset the chat and context with a button click.

## Technologies Used

- **Frontend**: React, JavaScript, CSS
- **Backend**: Node.js, Express
- **Data Fetching**: Fetch API for handling HTTP requests
- **Styling**: Custom CSS for UI components

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/react-pdf-viewer.git
    cd react-pdf-viewer
    ```

2. **Install frontend dependencies**:
    

3. **Install backend dependencies**:
    ```bash
    cd backend
    npm install
    ```

4. **Run the backend server**:
    ```bash
    node run dev
    ```

5. **Run the React application**:
    ```bash
    cd frontend
    npm start
    ```

The application will start, and you can access it via your web browser at `http://localhost:3000`.

## Project Structure

```
.
├── frontend
│   ├── public
        ├── final_response_log.json
│   ├── src
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
|   |   ├── ...
├── backend
│   ├── server.js
│   ├── ...
└── README.md
```

- **frontend/src/App.js**: Main React component containing the logic for rendering UI, handling user inputs, fetching data, and displaying PDFs.
- **frontend/src/App.css**: Styling for the React components.
- **backend/server.js**: Node.js server using Express to fetch PDFs from provided URLs and handle CORS issues.
- **backend/final_response_log.json**: JSON file containing the questions, answers, and source URLs for the application.

## Usage

1. Open the application in your browser.
2. Enter a question in the chat interface and click the submit button.
3. The application will check the JSON data for relevant information and fetch associated PDFs.
4. Click on context boxes to view the source PDFs in the embedded PDF viewer.

## How It Works

- The application loads data from `final_response_log.json` on startup.
- When a user inputs a question, the app checks if the question exists in the JSON data.
- If the question is valid, the app displays the question, answer, and clickable source boxes.
- Clicking on a source box fetches the PDF from the server using a URL and displays it using an embedded PDF viewer.
- The backend Express server handles fetching the actual PDF files from the URLs.
