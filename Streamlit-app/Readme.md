# Earnings Call Transcript Analysis

This project is a replication task developed for an internship. It involves analyzing earnings call transcripts using various NLP techniques and models. The goal is to extract, summarize, and categorize information effectively, with a focus on important topics and speaker-specific details.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project utilizes the LangChain framework along with OpenAI's language models to process and analyze earnings call transcripts. The primary objectives are to extract structured information, identify key topics, summarize speaker statements, classify the dialogue into questions & answers and a RAG chatbot is employed to provide interactive and contextually accurate responses to user queries. 

## Technologies Used

- **Streamlit**: A framework used to build interactive web applications.
- **LangChain**: A toolkit for building applications with large language models.
- **OpenAI (GPT-4o-mini)**: Language model for generating responses and extracting information.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **RAG (Retrieval-Augmented Generation) Chatbot**: A chatbot that combines retrieval and generation techniques to provide contextually accurate responses.
- **Python Libraries**: `re`, `pandas`, `dotenv` for regular expressions, data handling, and environment management, respectively.

## Features

1. **PDF Document Loading and Processing**:
   - Supports loading and reading PDF documents using `PyPDFLoader`.
   - Uses `TokenTextSplitter` for splitting text into manageable chunks.

2. **Information Extraction**:
   - Extracts company information, management details, and speaker-specific content using customized prompt templates.
   - Identifies and categorizes sentences as questions, answers, or general statements.

3. **Topic Identification and Summarization**:
   - Utilizes a `ChatPromptTemplate` to identify and cluster key topics from the transcripts.
   - Generates concise summaries of the identified topics, including numerical data where applicable.

4. **Speaker-Specific Text Extraction**:
   - Isolates and outputs all statements made by a specific speaker.
   - Supports multiple speakers, maintaining accurate association with the respective speaker.
5. **RAG Chatbot**:
   - Implements a Retrieval-Augmented Generation (RAG) chatbot to answer user queries based on the transcript content.
   - Uses FAISS to retrieve relevant document segments, providing contextually accurate and informative responses.

6. **Interactive User Interface**:
   - Built using Streamlit for an interactive and user-friendly experience.
   - Sidebars for navigation and buttons for specific tasks like processing the PDF, generating topics, and summarizing.

7. **Cost Calculation**:
   - Calculates token usage and cost estimation based on model usage.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/edithram23/genai/Streamlit-app
   ```

2. Navigate to the project directory:

   ```bash
   cd Streamlit-app
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file and add your OpenAI API key.

     ```text
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload a PDF file containing an earnings call transcript.

3. Use the navigation buttons to perform tasks such as extracting speaker-specific text, identifying key topics, and summarizing content.
