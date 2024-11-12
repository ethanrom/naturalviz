from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from memory import memory
from tools import zeroshot_tools
import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()





def read_first_3_rows():
    dataset_path = "dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        first_3_rows = df.head(3).to_string(index=False)
    except FileNotFoundError:
        first_3_rows = "Error: Dataset file not found."

    return first_3_rows


def get_agent_chain():


    dataset_first_3_rows = read_first_3_rows()

    prompt = PromptTemplate(

    input_variables = ['agent_scratchpad', 'chat_history', 'input'],
    template = (
        f"""
            You are a helpful assistant that can help users explore a dataset.
            First 3 rows of the dataset:
            {dataset_first_3_rows}


            Begin!
            """
            """
            chat history:
            {chat_history}

            New input: {input}
            {agent_scratchpad}"""
        ),
    )


    agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    agent = create_tool_calling_agent(agent_llm, tools, prompt)
    agent_memory=create_memory(chat_session_id)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=agent_memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    return agent_executor