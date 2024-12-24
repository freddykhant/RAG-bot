# adaptive local RAG agent with LLaMA3.2 🦙🔗

this project was me learning how retrieval augmented generation works by building a local RAG agent using llama 3.2 🚀
huge acknowledgements to LangChain and LangGraph tutorials and documentation

## about 🌟

the adaptive local RAG agent uses LLaMA3.2 to enhance the generation of responses by incorporating relevant information retrieved from a local knowledge base 📚

## features ✨

the RAG agent uses llama 3.2 has 3 components:

- **Routing**: Adaptive RAG to route questions to different retrieval approaches
- **Fallback**: Corrective RAG to fallback to web search if docs are not relevant to the query
- **Self Correction**: Self-RAG to fix irrelevant answers or ones with hallucinations

## installation 🛠️

installation steps:

1. **clone the repository**:

```bash
git clone https://github.com/yourusername/RAG-bot.git
cd RAG-bot
```

2. **install dependencies**:

```bash
pip install -r requirements.txt
```

## usage 🚀

to run the adaptive local RAG agent, use the following command:

```bash
python main.py
```
