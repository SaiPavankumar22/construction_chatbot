import os
import gradio as gr
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from typing import List, Tuple
import time

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY environment variable")
if not SERPER_API_KEY:
    raise ValueError("Missing SERPER_API_KEY environment variable")


crew_llm = LLM(
    model="openrouter/deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.7
)


direct_llm = ChatOpenAI(
    model="deepseek/deepseek-r1",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=2000
)

class ConstructionChatbot:
    def __init__(self):
        self.memory: List[Tuple[str, str]] = []  
        self.setup_tools()
        self.setup_crew()
    
    def setup_tools(self):
        """Set up web search tools"""
        try:
            self.search_tool = SerperDevTool()  
            print("✅ Web search tool initialized successfully")
        except Exception as e:
            self.search_tool = None
            print(f"⚠️  Warning: Could not initialize web search tool: {e}")
    
    def setup_crew(self):
        """Set up CrewAI agents and tasks"""
        # Tools list
        tools = []
        if self.search_tool:
            tools.append(self.search_tool)
        

        self.construction_agent = Agent(
            role='Construction Expert Assistant',
            goal='Provide accurate construction-related information ONLY. Reject all non-construction queries.',
            backstory="""You are a specialized construction industry expert with deep knowledge in:
            - Building safety and regulations
            - Fire safety codes and compliance
            - Construction materials and costs
            - Project management methodologies
            - Heavy machinery and equipment
            - Civil engineering principles
            - Structural design and analysis
            - Site management and safety protocols
            
            IMPORTANT: You MUST ONLY respond to construction-related questions. 
            If a user asks about anything not related to construction, building, 
            engineering, safety, materials, or project management, you must respond 
            with EXACTLY: "I can only assist with construction-related queries. Please ask about building, safety, materials, project management, or engineering topics."
            
            When you need current information about construction topics, use the search tool.""",
            llm=crew_llm,
            tools=tools,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=45
        )
        

        if self.search_tool:
            self.research_agent = Agent(
                role='Construction Research Specialist',
                goal='Search and gather current construction-related information from the internet ONLY',
                backstory="""You are a specialized researcher focused exclusively on construction industry topics.
                You search for the most current information about:
                - Construction practices and regulations
                - Building costs and material prices
                - Safety standards and compliance requirements
                - Industry trends and new technologies
                - Engineering standards and best practices
                
                You ONLY research construction-related topics. If asked to research non-construction 
                topics, decline politely and redirect to construction subjects.""",
                llm=crew_llm,
                tools=[self.search_tool],
                verbose=True,
                allow_delegation=False,
                max_iter=2,
                max_execution_time=30
            )
        else:
            self.research_agent = None
    
    def add_to_memory(self, user_query: str, response: str):
        """Add interaction to rolling memory window"""
        self.memory.append((user_query, response))
        if len(self.memory) > 5:
            self.memory.pop(0) 
    
    def get_chat_history(self) -> str:
        """Format chat history for prompt"""
        if not self.memory:
            return "No previous conversation."
        
        history = ""
        for i, (user_msg, bot_msg) in enumerate(self.memory, 1):
            history += f"Message {i}:\nUser: {user_msg}\nAssistant: {bot_msg}\n\n"
        return history.strip()
    
    def is_construction_related(self, query: str) -> bool:
        """Simple check if query is construction-related"""
        construction_keywords = [
            'construction', 'building', 'concrete', 'steel', 'foundation', 'safety',
            'project management', 'engineering', 'structure', 'material', 'cost',
            'regulation', 'fire safety', 'osha', 'machinery', 'equipment', 'site',
            'contractor', 'cement', 'rebar', 'excavation', 'blueprint', 'architect',
            'electrical', 'plumbing', 'hvac', 'roofing', 'insulation', 'drywall'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in construction_keywords)
    
    def generate_response_with_crew(self, user_query: str) -> str:
        """Generate response using CrewAI with web search capabilities"""

        if not self.is_construction_related(user_query):
            response = "I can only assist with construction-related queries. Please ask about building, safety, materials, project management, or engineering topics."
            self.add_to_memory(user_query, response)
            return response
        
        chat_history = self.get_chat_history()
        
        try:

            search_keywords = ['current', 'latest', 'recent', 'today', '2024', '2025', 'price', 'cost', 'regulation', 'new', 'trend']
            needs_search = any(keyword in user_query.lower() for keyword in search_keywords)
            
            if needs_search and self.research_agent:

                research_task = Task(
                    description=f"""Search for current construction-related information about: {user_query}
                    
                    Focus on finding:
                    - Latest construction industry data
                    - Current material prices and costs
                    - Recent regulations and safety updates
                    - New construction technologies and methods
                    - Industry trends and market information
                    
                    Search query should be concise and focused on construction industry information.
                    """,
                    expected_output="Current, accurate construction industry information and data",
                    agent=self.research_agent
                )
                

                response_task = Task(
                    description=f"""Based on research findings and chat history, provide a comprehensive response to: {user_query}
                    
                    Chat history: {chat_history}
                    
                    Guidelines:
                    - Use the research data to provide accurate, current information
                    - Focus on construction industry expertise
                    - Provide practical, actionable advice
                    - Include specific details like prices, regulations, or technical specifications when available
                    - Structure the response clearly and professionally
                    """,
                    expected_output="Detailed, informative construction industry response with current data",
                    agent=self.construction_agent,
                    context=[research_task]
                )
                

                crew = Crew(
                    agents=[self.research_agent, self.construction_agent],
                    tasks=[research_task, response_task],
                    verbose=False 
                )
                
            else:

                response_task = Task(
                    description=f"""Provide expert construction advice for: {user_query}
                    
                    Chat history: {chat_history}
                    
                    Guidelines:
                    - Draw from your construction industry expertise
                    - Provide detailed, accurate information
                    - Include relevant safety considerations
                    - Suggest best practices and standards
                    - Structure the response professionally
                    """,
                    expected_output="Expert construction industry advice and information",
                    agent=self.construction_agent
                )
                

                crew = Crew(
                    agents=[self.construction_agent],
                    tasks=[response_task],
                    verbose=False
                )
            

            result = crew.kickoff()
            response = str(result).strip()
            

            if not response or len(response) < 10:
                response = "I apologize, but I'm having trouble generating a proper response. Could you please rephrase your construction-related question?"
            

            self.add_to_memory(user_query, response)
            return response
            
        except Exception as e:
            print(f"CrewAI Error: {e}")

            return self.generate_response_direct(user_query)
    
    def generate_response_direct(self, user_query: str) -> str:
        """Fallback method using direct LLM with construction filtering"""
        if not self.is_construction_related(user_query):
            response = "I can only assist with construction-related queries. Please ask about building, safety, materials, project management, or engineering topics."
            self.add_to_memory(user_query, response)
            return response
        
        chat_history = self.get_chat_history()
        
        prompt = f"""You are a specialized construction industry AI assistant with expertise in building, safety, materials, project management, and engineering.

Chat history: {chat_history}

User question: {user_query}

Provide a detailed, professional response focusing on construction industry knowledge. Include specific information about safety standards, building codes, material specifications, cost estimates, or project management advice as relevant to the question.

Response:"""

        try:
            response = direct_llm.invoke(prompt)
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            self.add_to_memory(user_query, response_text)
            return response_text
            
        except Exception as e:
            fallback_response = f"""I apologize, but I'm experiencing technical difficulties. However, I can still help with construction-related questions about safety, materials, project management, and engineering. Please try rephrasing your question.

Technical error: {str(e)[:100]}..."""
            
            self.add_to_memory(user_query, fallback_response)
            return fallback_response
    
    def generate_response(self, user_query: str) -> str:
        """Main response generation method"""
        try:
            return self.generate_response_with_crew(user_query)
        except Exception as e:
            print(f"Crew method failed, using direct method: {e}")
            return self.generate_response_direct(user_query)


chatbot = ConstructionChatbot()

custom_css = """
/* Construction-themed color palette */
:root {
    --construction-orange: #ff6b1a;
    --construction-yellow: #ffc107;
    --construction-blue: #1e88e5;
    --construction-green: #4caf50;
    --construction-red: #f44336;
    --construction-dark: #2d3436;
    --construction-darker: #1a1a1a;
    --construction-light-gray: #ecf0f1;
    --construction-medium-gray: #95a5a6;
    --construction-steel: #34495e;
}

/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, var(--construction-dark) 0%, var(--construction-darker) 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

/* Header styling */
.construction-header {
    background: linear-gradient(45deg, var(--construction-orange), var(--construction-yellow));
    color: #000;
    text-align: center;
    padding: 25px;
    font-weight: bold;
    font-size: 28px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(255, 107, 26, 0.4);
    border: 3px solid var(--construction-yellow);
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* Chat container - Custom styling */
.chatbot {
    border: 3px solid var(--construction-orange) !important;
    border-radius: 15px !important;
    background: linear-gradient(135deg, var(--construction-steel), #2c3e50) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

/* Message bubbles styling */
.message-wrap {
    margin: 15px 0 !important;
}

.message-wrap.svelte-1lcyrx4.user {
    justify-content: flex-end !important;
}

.message-wrap.svelte-1lcyrx4.bot {
    justify-content: flex-start !important;
}

/* User message styling */
.message-wrap.user .message {
    background: linear-gradient(135deg, var(--construction-orange), var(--construction-yellow)) !important;
    color: #000 !important;
    border-radius: 20px 20px 5px 20px !important;
    padding: 15px 20px !important;
    max-width: 75% !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(255, 107, 26, 0.3) !important;
    border: 2px solid var(--construction-yellow) !important;
}

/* Bot message styling */
.message-wrap.bot .message {
    background: linear-gradient(135deg, var(--construction-blue), #1565c0) !important;
    color: #ffffff !important;
    border-radius: 20px 20px 20px 5px !important;
    padding: 15px 20px !important;
    max-width: 75% !important;
    box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3) !important;
    border-left: 5px solid var(--construction-green) !important;
}

/* Input styling */
.input-container {
    background: linear-gradient(135deg, var(--construction-steel), #2c3e50);
    border-radius: 25px;
    padding: 15px;
    border: 3px solid var(--construction-orange);
    margin-top: 20px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
}

/* Textbox styling */
.textbox {
    background: var(--construction-dark) !important;
    border: 2px solid var(--construction-orange) !important;
    border-radius: 20px !important;
    color: var(--construction-light-gray) !important;
    padding: 15px 20px !important;
    font-size: 16px !important;
}

.textbox:focus {
    border-color: var(--construction-yellow) !important;
    box-shadow: 0 0 15px rgba(255, 193, 7, 0.3) !important;
}

/* Button styling */
.btn-primary {
    background: linear-gradient(135deg, var(--construction-orange), var(--construction-yellow)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 15px 25px !important;
    font-weight: bold !important;
    font-size: 16px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(255, 107, 26, 0.3) !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, var(--construction-yellow), #ffeb3b) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 193, 7, 0.4) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, var(--construction-steel), #34495e) !important;
    color: var(--construction-light-gray) !important;
    border: 2px solid var(--construction-medium-gray) !important;
    border-radius: 15px !important;
    padding: 12px 20px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.btn-secondary:hover {
    background: linear-gradient(135deg, var(--construction-blue), #1976d2) !important;
    border-color: var(--construction-blue) !important;
    transform: translateY(-2px) !important;
}

/* Status and info styling */
.status-info {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(30, 136, 229, 0.1));
    border: 2px solid var(--construction-green);
    border-radius: 12px;
    padding: 15px;
    margin: 15px 0;
    color: var(--construction-light-gray);
    font-size: 14px;
    box-shadow: 0 3px 10px rgba(76, 175, 80, 0.2);
}

/* Warning/info boxes */
.warning-box {
    background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(255, 107, 26, 0.1));
    border: 2px solid var(--construction-red);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    color: var(--construction-light-gray);
    box-shadow: 0 4px 15px rgba(244, 67, 54, 0.2);
}

.info-box {
    background: linear-gradient(135deg, rgba(30, 136, 229, 0.1), rgba(255, 193, 7, 0.1));
    border: 2px solid var(--construction-blue);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    color: var(--construction-light-gray);
    box-shadow: 0 4px 15px rgba(30, 136, 229, 0.2);
}

.success-box {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(255, 193, 7, 0.1));
    border: 2px solid var(--construction-green);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    color: var(--construction-light-gray);
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--construction-orange) !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
    font-weight: bold !important;
}

/* Links */
a {
    color: var(--construction-yellow) !important;
    text-decoration: none !important;
    font-weight: bold !important;
}

a:hover {
    color: var(--construction-orange) !important;
    text-decoration: underline !important;
}

/* Code styling */
code {
    background: var(--construction-darker) !important;
    color: var(--construction-yellow) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
}

/* Lists */
ul, ol {
    color: var(--construction-light-gray);
}

li {
    margin: 8px 0;
    padding-left: 5px;
    border-left: 3px solid var(--construction-orange);
    padding-left: 10px;
    margin-left: 10px;
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 10px;
    animation: pulse 2s infinite;
}

.status-online {
    background: var(--construction-green);
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.6);
}

.status-offline {
    background: var(--construction-red);
    box-shadow: 0 0 15px rgba(244, 67, 54, 0.6);
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Badge styling */
.badge {
    background: linear-gradient(135deg, var(--construction-blue), #1565c0);
    color: white;
    padding: 6px 12px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    display: inline-block;
    margin: 2px;
    box-shadow: 0 2px 8px rgba(30, 136, 229, 0.3);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--construction-darker);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, var(--construction-orange), var(--construction-yellow));
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, var(--construction-yellow), #ffeb3b);
}

/* Responsive design */
@media (max-width: 768px) {
    .construction-header {
        font-size: 20px;
        padding: 20px 15px;
    }
    
    .message {
        max-width: 90% !important;
        padding: 12px 15px !important;
    }
}

/* Animation for new messages */
.message-wrap {
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
"""

def create_interface():
    """Create the Gradio interface with construction theme"""
    
    with gr.Blocks(css=custom_css, title="🏗️ Construction AI Assistant - DeepSeek R1 🏗️") as interface:
        

        gr.HTML("""
        <div class="construction-header">
            🏗️ CONSTRUCTION COMPANY AI ASSISTANT 🏗️
            <br>
            <small style="font-size: 18px; font-weight: normal; opacity: 0.9;">
                Powered by <span class="badge">DeepSeek R1</span> | Expert Construction Guidance with Live Web Search
            </small>
        </div>
        """)
        

        def get_status_info():
            search_status = "🌐 Web Search: ONLINE" if chatbot.search_tool else "🌐 Web Search: OFFLINE"
            search_class = "status-online" if chatbot.search_tool else "status-offline"
            memory_count = len(chatbot.memory)
            
            return f"""
            <div class="status-info">
                <span class="status-indicator {search_class}"></span>
                <strong>🧠 DeepSeek R1 ACTIVE</strong> | 
                <strong>{search_status}</strong> | 
                <strong>💭 Memory: {memory_count}/5</strong> | 
                <strong>🏗️ Construction Expert Ready</strong>
                <br>
                <small>🔍 Real-time construction data | 📊 Current prices & regulations | ⚡ Lightning fast responses</small>
            </div>
            """
        
        memory_status = gr.HTML(get_status_info())
        

        with gr.Row():
            with gr.Column():
                chatbot_interface = gr.Chatbot(
                    [],
                    height=500,
                    show_label=False,
                    container=True,
                    avatar_images=("👤", "🏗️"),
                    bubble_full_width=False,
                    type="messages",
                    show_copy_button=True,
                    elem_classes=["chatbot"]
                )
        

        with gr.Row():
            with gr.Column(scale=8):
                msg_input = gr.Textbox(
                    placeholder="🏗️ Ask me about construction, safety, materials, project management, engineering, costs, regulations... (Non-construction queries will be declined)",
                    show_label=False,
                    container=False,
                    lines=2,
                    max_lines=4,
                    elem_classes=["textbox"]
                )
            with gr.Column(scale=2, min_width=120):
                send_btn = gr.Button(
                    "🚀 SEND", 
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-primary"]
                )
        
       
        with gr.Row():
            with gr.Column(scale=1):
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", elem_classes=["btn-secondary"])
            with gr.Column(scale=1):
                examples_btn = gr.Button("💡 Examples", variant="secondary", elem_classes=["btn-secondary"])
        
      
        examples_html = f"""
        <div class="success-box">
            <h4>✅ CONSTRUCTION QUESTIONS I CAN ANSWER {'(with Live Web Search!)' if chatbot.search_tool else ''}:</h4>
            <ul>
                <li><strong>📊 Current Costs:</strong> "What are today's steel and concrete prices for construction?"</li>
                <li><strong>🔥 Safety Regulations:</strong> "Latest fire safety codes for high-rise buildings in 2024?"</li>
                <li><strong>🏗️ Engineering:</strong> "How to calculate foundation requirements for a 10-story building?"</li>
                <li><strong>⚠️ Safety Equipment:</strong> "OSHA requirements for construction site safety equipment?"</li>
                <li><strong>📋 Project Management:</strong> "Best methodologies for managing large construction projects?"</li>
                <li><strong>🌱 Sustainable Materials:</strong> "Latest trends in eco-friendly construction materials?"</li>
                <li><strong>🔧 Equipment:</strong> "Heavy machinery recommendations for excavation projects?"</li>
                <li><strong>📐 Structural Design:</strong> "Load-bearing calculations for steel frame buildings?"</li>
            </ul>
        </div>
        
        <div class="warning-box">
            <h4>❌ NON-CONSTRUCTION TOPICS (Will Be Politely Declined):</h4>
            <ul>
                <li>Sports, entertainment, cooking, general knowledge</li>
                <li>Programming, medicine, finance (unless construction-related)</li>
                <li>Personal advice, relationship help</li>
                <li>Any topic not related to construction, building, or engineering</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>💡 PRO TIPS FOR BEST RESULTS:</h4>
            <ul>
                <li><strong>🔍 Trigger Web Search:</strong> Use words like "current", "latest", "2024", "price", "cost" for real-time data</li>
                <li><strong>📍 Be Specific:</strong> Mention location, building type, or specific materials for targeted advice</li>
                <li><strong>📋 Multiple Questions:</strong> I remember our last 5 conversations for context</li>
                <li><strong>🎯 Focus:</strong> Stay within construction topics for the best expert guidance</li>
            </ul>
        </div>
        """
        
        examples_display = gr.HTML(examples_html, visible=False)
        

        def update_memory_display():
            return get_status_info()
        
        def respond(message, history):
            if not message.strip():
                return history, "", update_memory_display()
            

            if history is None:
                history = []
            
       
            history.append({"role": "user", "content": message})
            

            try:
                response = chatbot.generate_response(message)
                history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_response = f"🚨 I apologize, but I encountered an error while processing your construction query. Please try again or rephrase your question.\n\nError: {str(e)[:100]}..."
                history.append({"role": "assistant", "content": error_response})
            
            return history, "", update_memory_display()
        
        def clear_chat():
            chatbot.memory.clear()
            return [], update_memory_display()
        
        def toggle_examples(current_visibility):
            return not current_visibility
        

        msg_input.submit(respond, [msg_input, chatbot_interface], [chatbot_interface, msg_input, memory_status])
        send_btn.click(respond, [msg_input, chatbot_interface], [chatbot_interface, msg_input, memory_status])
        clear_btn.click(clear_chat, outputs=[chatbot_interface, memory_status])
        examples_btn.click(toggle_examples, inputs=examples_display, outputs=examples_display)
        
 
        if chatbot.search_tool:
            gr.HTML(f"""
            <div class="success-box">
                <h4>🌐 WEB SEARCH ENABLED - REAL-TIME CONSTRUCTION DATA!</h4>
                <p><strong>✅ Connected to Serper API</strong> - I can fetch the latest construction industry information including:</p>
                <ul>
                    <li>Current material prices and market trends</li>
                    <li>Latest building codes and regulations</li>
                    <li>New construction technologies and methods</li>
                    <li>Industry news and safety updates</li>
                    <li>Equipment specifications and pricing</li>
                </ul>
                <p><small>💡 <strong>Tip:</strong> Ask questions with "current", "latest", "2024", "price" to automatically trigger web search!</small></p>
            </div>
            """)
        else:
            gr.HTML(f"""
            <div class="warning-box">
                <h4>⚠️ WEB SEARCH CURRENTLY DISABLED</h4>
                <p><strong>Reason:</strong> Serper API key not properly configured</p>
                <p><strong>Impact:</strong> I can still answer construction questions using my knowledge base, but won't have access to real-time data.</p>
                
                <h5>🔧 To Enable Web Search:</h5>
                <ol>
                    <li>Visit <a href="https://serper.dev" target="_blank">serper.dev</a> and get a free API key</li>
                    <li>Set environment variable: <code>export SERPER_API_KEY='your-key-here'</code></li>
                    <li>Restart the application</li>
                </ol>
                <p><small>💡 Even without web search, I'm still a comprehensive construction expert ready to help!</small></p>
            </div>
            """)
        
   
        gr.HTML(f"""
        <div class="info-box">
            <h4>🧠 POWERED BY DEEPSEEK R1 - ADVANCED REASONING AI</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
                <div>
                    <h5>🔬 Model Capabilities:</h5>
                    <ul>
                        <li>Advanced reasoning and problem-solving</li>
                        <li>Deep construction industry knowledge</li>
                        <li>Complex calculation capabilities</li>
                        <li>Multi-step project planning</li>
                    </ul>
                </div>
                <div>
                    <h5>🎯 Specialized Features:</h5>
                    <ul>
                        <li>Construction-only query filtering</li>
                        <li>Memory of last 5 conversations</li>
                        <li>Web search integration for current data</li>
                        <li>Professional construction advice</li>
                    </ul>
                </div>
                <div>
                    <h5>🛡️ Safety & Compliance:</h5>
                    <ul>
                        <li>OSHA regulation expertise</li>
                        <li>Building code compliance</li>
                        <li>Fire safety standards</li>
                        <li>Risk assessment guidance</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
  
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(135deg, rgba(255, 107, 26, 0.1), rgba(255, 193, 7, 0.1)); border-radius: 15px; border: 2px solid var(--construction-orange);">
            <h5 style="color: var(--construction-orange); margin-bottom: 10px;">🏗️ PROFESSIONAL CONSTRUCTION AI ASSISTANT 🏗️</h5>
            <p style="color: var(--construction-light-gray); margin: 0;">
                <strong>Specialized in:</strong> Building Safety • Material Costs • Project Management • Engineering • Regulations • Equipment
                <br>
                <small>⚡ Powered by DeepSeek R1 via OpenRouter • 🌐 Enhanced with Web Search • 🎯 Construction-Focused AI</small>
            </p>
        </div>
        """)
    
    return interface


if __name__ == "__main__":
    print("🏗️" + "="*80)
    print("🏗️  STARTING CONSTRUCTION COMPANY AI ASSISTANT")
    print("🏗️" + "="*80)
    print("🧠  AI Model: DeepSeek R1 (Advanced Reasoning)")
    print("🌐  API Provider: OpenRouter")
    print("🔍  Web Search: Serper API")
    print(f"📡  Search Status: {'✅ ENABLED' if chatbot.search_tool else '❌ DISABLED (Check API Key)'}")
    print("🤖  Agent Framework: CrewAI")
    print("💬  Interface: Gradio with Construction Theme")
    print("🚨  Query Filter: Construction Topics ONLY")
    print("💭  Memory: Rolling window of 5 conversations")
    print("🏗️" + "="*80)
    print("🚀  Launching interface...")
    

    if chatbot.search_tool:
        print("✅  Web search tool initialized successfully!")
        print("💡  Users can get real-time construction data")
    else:
        print("⚠️   Web search disabled - check SERPER_API_KEY")
        print("💡  Assistant will work with knowledge base only")
    
    print("🏗️" + "="*80)
    

    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7863,
        show_error=True,
        debug=False,
        favicon_path=None,
        inbrowser=True
    )