
# Project Title

A brief description of what this project does and who it's for

# Medbot - A Disease Prediction Chatbot

Medbot is an intelligent chatbot designed to assist users in predicting diseases and answering health-related queries. It leverages state-of-the-art AI models, vector databases, and APIs to provide accurate and context-aware responses.

## Features
- **Flask-Based Web Application**: Runs a lightweight server for user interactions.
- **Natural Language Processing**: Utilizes LangChain and Sentence Transformers for handling user input.
- **Google Gemini Integration**: Employs advanced generative AI for context refinement and generating responses.
- **Vector Database**: Uses Pinecone for storing and retrieving contextual data.
- **Interactive Memory**: Maintains conversation history for a personalized chat experience.

## Prerequisites
Before setting up the project, ensure you have the following installed:
- Python 3.8 or higher
- Required Python packages (see [Dependencies](#dependencies))
- Pinecone and Google Gemini API keys

## Dependencies
Install the following Python packages using pip:

```bash
pip install flask psycopg2 flask-cors langchain langchain-core sentence-transformers pinecone-client google-generativeai
```
## Prerequisites
## Project Structure
final.py: Main application file containing the Flask server and core chatbot logic.

new.html: HTML template for the chatbot user interface.

data/: Directory containing documents for vector database initialization.

# Setup Instructions
1. Clone the Repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
2. Install Dependencies: Run the following command to install all required Python packages:
```bash
pip install -r requirements.txt
```
3. Set API Keys: Add your API keys in the appropriate sections of final.py:

- Replace your-pinecone-api-key with your Pinecone API key.
- Replace the placeholder Google Gemini API key with your key.
4. Initialize Vector Database: Prepare your document data and run the vector database setup script:
```bash
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
# Add database initialization code as provided in the project file
```
5. Run the Application: Start the Flask server using:
```bash
python final.py
```
## Usage
- Navigate to http://localhost:5000 in your browser.
- Type your query in the chat interface.
- Receive accurate and context-aware responses.
# Key Functionalities
## Chatbot Features
- User Interaction: Captures user input and maintains session history.
- Query Refinement: Refines queries using Google Gemini for improved response  accuracy.
- Contextual Search: Finds relevant context from Pinecone's vector database.
- Response Generation: Generates responses based on context and refined queries.
## Vector Database
- Initializes and stores document embeddings for similarity-based search.
- Allows retrieving the most relevant documents for user queries.
## Alternative Questions
- Provides users with three related alternative questions to enhance understanding.
## Future Enhancements
- Expand the dataset to include more health-related knowledge.
- Add support for multilingual queries.
- Implement real-time health diagnostics.

## Acknowledgments
- LangChain for conversational AI tools.
- Sentence Transformers for natural language embeddings.
- Google Gemini for generative AI capabilities.
- Pinecone for vector-based similarity search.