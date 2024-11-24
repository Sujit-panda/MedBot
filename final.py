from flask import Flask, request, jsonify, render_template
from langchain_core.runnables import RunnableWithMessageHistory, Runnable  # Updated based on LangChain deprecation
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set Google Gemini and Pinecone API keys
genai.configure(api_key="AIzaSyAqPZGrv4ZELWiEt0spuY-od33-GMSk7DY")
pinecone_api_key = 'your-pinecone-api-key'

# Initialize Pinecone
pc = Pinecone(api_key="f8c65da6-af5b-407b-8aa5-6ba3cc89af91")
index = pc.Index('project')  # Initialize Pinecone index for querying

# Initialize SentenceTransformer model for encoding user input
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize conversation memory buffer
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# System message template for guiding the model's behavior
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I DON'T KNOW'."""
)

# Templates for handling human input
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Custom Runnable for wrapping Google Gemini LLM
class GeminiRunnable(Runnable):
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def invoke(self, input_text):
        # Implement the invoke method to handle the input and output
        response = self.gemini_model.generate_content(input_text)
        return response.text.strip()

# Instantiate the Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Wrap the Gemini model in the custom runnable
gemini_runnable = GeminiRunnable(gemini_model)

# Initialize conversation using RunnableWithMessageHistory
conversation = RunnableWithMessageHistory(
    runnable=gemini_runnable,  # Use the custom Gemini Runnable
    get_session_history=lambda: buffer_memory.load_memory_variables({}),
    memory=buffer_memory,  # Conversation history
    prompt=prompt_template,  # Prompt template for guiding conversation
    verbose=True  # For logging
)

# Session state simulation to track conversation history
session_state = {
    'responses': ["Hello, Welcome to our chatBot!"],
    'requests': []
}

# Helper function to find matches from Pinecone index
def find_match(input):
    try:
        input_em = model.encode(input).tolist()
        result = index.query(vector=input_em, top_k=2, includeMetadata=True)
        if 'matches' in result and len(result['matches']) >= 2:
            return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
        else:
            return "No sufficient matches found in the index."
    except Exception as e:
        print(f"Error in find_match: {e}")
        return "An error occurred while querying the index."

# Refine user query to better fit the context using Google Gemini
def query_refiner(conversation, query):
    response = gemini_model.generate_content(f"Refine the following user query based on the conversation:\n\nUser Query: {query}\nConversation: {conversation}")
    refined_query = response.text.strip()
    return refined_query

# Construct the conversation string from session state
def get_conversation_string(responses, requests):
    conversation_string = ""
    for i in range(len(responses) - 1):
        conversation_string += "Human: " + requests[i] + "\n"
        conversation_string += "Bot: " + responses[i + 1] + "\n"
    return conversation_string

# Generate alternative questions based on user input using Google Gemini
def generate_alternative_questions(query):
    prompt = f"Provide 3 alternative questions related to '{query}' in simple English."
    response = gemini_model.generate_content(prompt)
    alternatives = response.text.strip().split('\n')
    return [alt.strip() for alt in alternatives if alt.strip()]

# Route to serve the main HTML page
@app.route('/')
def hello():
    return render_template('new.html')

# Route to handle chat messages from the user
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message')
    print("User Message:", user_message)

    # Update session state with the user's message
    session_state['requests'].append(user_message)
    conversation_string = get_conversation_string(session_state['responses'], session_state['requests'])
    print("Conversation String:", conversation_string)

    # Refine the query based on the conversation
    refined_query = query_refiner(conversation_string, user_message)
    print("Refined Query:", refined_query)

    # Find matching context from Pinecone
    context = find_match(refined_query)
    print("Context:", context)

    # Get the chatbot's response based on the context and user's query
    response = gemini_model.generate_content(f"Context:\n{context}\n\nQuery:\n{user_message}").text
    print("Response:", response)

    # Generate alternative questions based on the refined query
    alternatives = generate_alternative_questions(refined_query)

    # Update session state with the bot's response
    session_state['responses'].append(response)

    # Return the response and related questions as a JSON object
    return jsonify({
        "response": response,
        "refined_query": refined_query,
        "related_questions": alternatives
    })

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
