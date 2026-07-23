Building Applications with AI
Overview
Architecture of a domain-specialised RAG chatbot with tool calling
Preview

RAG chatbot flow: vector retrieval, LLM reasoning and tool calling

This is the project for Sprint 2, where you bring together LangChain, RAG, function calling, and vector databases to build a specialised, domain-focused chatbot. Unlike the Sprint 1 project, this one requires advanced RAG with query translation, tool calling for practical tasks, and a polished user interface. The result should be something that provides real value in a specific domain, and the architectural patterns you use here will carry directly into Sprint 3's autonomous agents.
Topics

    Advanced RAG Implementation
    Tool Calling
    ChatGPT
    Prompt Engineering
    Streamlit / Next.js
    LangChain
    Vector Databases

Prerequisites

    Python / TypeScript knowledge
    Knowledge of ChatGPT and the OpenAI SDK (used here via OpenRouter)
    Basic knowledge of Streamlit / Next.js
    Understanding of RAG concepts from Parts 2 and 3
    Familiarity with tool calling

Estimated time to complete: 20-25 hours
Table of contents

    Task description
    Task requirements
        Core requirements
    Optional tasks
        Easy
        Medium
        Hard
    Evaluation criteria
    Creating the web app
    Approach to solving the task
    Submission
        Submission and scheduling a project review
    Additional resources

Task description

This project represents a significant step up from basic chatbot implementation. You will create a specialised chatbot that leverages advanced RAG techniques and tool calling to provide domain-specific assistance. The goal is to build something that could be valuable in a real-world context.

We will be using LangChain, your framework (Streamlit or Next.js) of choice, and implementing advanced RAG techniques.

The intended code editor for this project is VS Code.

Your chatbot will:

    Implement advanced RAG with query translation and structured retrieval
    Include tool calling capabilities for practical tasks
    Focus on a specific domain or use case
    Provide detailed, context-aware responses

Example use cases (but feel free to create your own):

    Career Consultant Bot: Uses World Economic Forum reports and job market data to provide career advice
    Technical Documentation Assistant: Helps developers understand and work with specific frameworks or libraries
    Financial Advisor Bot: Analyses market trends and provides investment insights
    Healthcare Information Assistant: Provides accurate medical information from verified sources
    Legal Research Assistant: Helps with legal queries using case law and legal documents

Task requirements
Core requirements

    RAG Implementation:
        Create a knowledge base relevant to your domain
        Implement standard document retrieval with embeddings
        Use chunking strategies and similarity search

    Tool Calling:
        Implement at least 3 different tool calls
        Functions should be relevant to your domain
        Examples: data analysis, calculations, API integrations

    Domain Specialisation:
        Choose a specific domain or use case
        Create a focused knowledge base
        Implement domain-specific prompts and responses
        Add relevant security measures for your domain

    Technical Implementation:
        Use LangChain with OpenRouter (OpenAI-compatible SDK) for LLM integration
        Implement proper error handling
        Include user input validation

    User Interface:
        Create an intuitive interface using Streamlit or Next.js
        Show relevant context and sources
        Display tool call results
        Include progress indicators for long operations

Optional tasks

After the main functionality is implemented and your code works correctly, and you feel that you want to upgrade your project, choose one or more improvements from this list. The list is sorted by difficulty level.

Caution: Some of the tasks in medium or hard categories may involve concepts or libraries that are introduced in later sections or even require outside knowledge/time to research outside of the course.
Easy

    Add conversation history and export functionality
    Add visualisation of RAG process
    Include source citations in responses
    Add an interactive help feature or chatbot guide

Medium

    Implement multi-model support (OpenAI, Anthropic, etc.)
    Add real-time data updates to knowledge base
    Protect your app against prompt injection
    Add user authentication and personalisation
    Calculate and display token usage and costs
    Add visualisation of tool call results
    Implement conversation export in various formats (PDF, CSV, JSON)
    Connect to tools from a publicly available remote MCP server
    Implement rate limiting and API key management
    Add logging and monitoring

Hard

    Employ hybrid search
    Implement A/B testing for different RAG strategies
    Add automated knowledge base updates
    Add multi-language support
    Implement advanced analytics dashboard
    Implement your tools (functions) as MCP servers
    Implement an evaluation of your RAG system, using RAGAs or otherwise

Evaluation criteria

    Understanding Core Concepts:
        The learner understands the basic principles of how RAG works
        The learner can explain tool calling implementation clearly
        The learner demonstrates good code organisation practices
        The learner can identify potential error scenarios and edge cases

    Technical Implementation:
        The learner knows how to use a front-end library using their knowledge and/or external resources
        The learner's project works as intended; you can chat with a chatbot and get answers
        The learner has created a relevant knowledge base for their domain
        The learner has implemented appropriate security considerations

    Reflection and Improvement:
        The learner understands the potential problems with the application
        The learner can offer suggestions on improving the code and the project

    Bonus Points:
        For maximum points, the learner should implement at least 2 medium and 1 hard optional task.

Creating the web app

There are a number of ways we can choose to develop our application; here's some information about a few frameworks we recommend:
Python track: Streamlit

How to get started with Streamlit

It is very likely that you are seeing and hearing about Streamlit for the first time. No worries!

It's a fantastic framework for creating interactive web apps using Python, particularly for data visualization, machine learning demos, and quick prototyping.

You don't need to know much about front-end things, like HTML, CSS, JS, React, or others, to build apps! Streamlit will do the basics of front-end for you by just writing Python code.

Learning Streamlit:

    You can get started by watching this video.
    After that, check out their page.
    Check their documentation on page elements.
    A good starting point could be their "Get Started" section.
    Lastly, GeeksForGeeks also offers a good tutorial on Streamlit.
    YouTube short.
    Tutorial on using Streamlit in VS Code.

JavaScript track: Next.js

Approach to solving the task

    1-5 hours of attempting to solve the task using your own knowledge + ChatGPT. It is encouraged to use ChatGPT both for:
        Understanding this task better
        Writing the code
        Improving the code
        Understanding the code
    You can also take a look at the "How to Get Started" sections to better understand how to get started with Streamlit or Next.js.
    If you feel that some knowledge is missing, please revisit the parts in Sprint 1 OR check out additional resources.
    If during the first 1-2 hours you see you are making no progress and the task seems much too hard for you, we recommend spending 10 more hours working on the problem with help from peers and JTLs.
