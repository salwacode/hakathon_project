from flask import Flask, render_template, request, session
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

app = Flask(__name__)
#replace this with your api key
API_KEY = 'open_api_key'

teacher_template = """As a teacher bot, I'm here to ask some questions to understand your student's level in a specific subject. 

{history}
Teacher: {human_input}
Chatbot:"""

student_template = """As a student bot, I'm here to assist you with any topics you are struggling to understand. 

{history}
Student: {human_input}
Chatbot:"""

teacher_prompt = PromptTemplate(input_variables=["history", "human_input"], template=teacher_template)
student_prompt = PromptTemplate(input_variables=["history", "human_input"], template=student_template)

teacher_chain = LLMChain(
    llm=OpenAI(openai_api_key=API_KEY, temperature=0),
    prompt=teacher_prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

student_chain = LLMChain(
    llm=OpenAI(openai_api_key=API_KEY, temperature=0),
    prompt=student_prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['user_input']

        if 'teacher_bot_history' not in session:
            session['teacher_bot_history'] = ""
            session['student_bot_history'] = ""

        # Teacher bot
        teacher_output = teacher_chain.predict(
            history=session['teacher_bot_history'],
            human_input=user_input
        )
        session['teacher_bot_history'] += f"Teacher: {user_input}\nChatbot: {teacher_output}\n"

        # Student bot
        student_output = student_chain.predict(
            history=session['student_bot_history'],
            human_input=user_input
        )
        session['student_bot_history'] += f"Student: {user_input}\nChatbot: {student_output}\n"

        return {
            'teacher_output': teacher_output,
            'student_output': student_output
        }

    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'
    app.run(debug=True)