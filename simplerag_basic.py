from crewai import Agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
)

def create_simple_agent():
    # Create a basic question-answering agent
    agent = Agent(
        role='Question Answerer',
        goal='Answer questions clearly and concisely',
        backstory="""You are a helpful AI assistant that answers questions directly and accurately. 
        Keep responses clear and to the point.""",
        llm=llm,
        verbose=True
    )
    return agent

if __name__ == "__main__":
    agent = create_simple_agent()
    test_result = agent.llm.call("Tell me a lame dad joke")
    print(test_result)
    
    test_result = llm.invoke("Tell me a good dad joke")
    print(test_result.content)
