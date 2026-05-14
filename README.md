# AI Agent Evaluation & Benchmarking
[![📄 Full Project Report](https://img.shields.io/badge/READ_FULL_PROJECT_REPORT-2563EB?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://michelin-ensimag.github.io/AI-Agent-Testing/report)

[![CI](https://img.shields.io/github/actions/workflow/status/Michelin-Ensimag/AI-Agent-Testing/ci.yml?branch=main&style=flat-square)](https://github.com/Michelin-Ensimag/AI-Agent-Testing/actions/workflows/ci.yml) 
[![DeepEval Results](https://img.shields.io/website?url=https%3A%2F%2Fmichelin-ensimag.github.io%2FAI-Agent-Testing%2Fdeepeval_results%2F&up_message=live&down_message=offline&label=DeepEval%20Results&style=flat-square)](https://michelin-ensimag.github.io/AI-Agent-Testing/deepeval_results/)

An evaluation toolkit designed to measure AI agent robustness. Contains automated testing pipelines for internal agents and a benchmarking methodology to audit multi-agent architectures for complex reasoning flaws.

## Project Context
This project was developed as a **Specialty Project (2026)** in a partnership between **Michelin** and **Grenoble INP - Ensimag**. It focuses on the rise of multi-agent architectures and the **Model Context Protocol (MCP)**, aiming to provide an objective framework for measuring agent reliability in industrial contexts.

## Project Scope & Methodology
Our work is structured across three phases:

### Phase 1: Custom Agent Orchestration
We explored agent orchestration mechanisms by building several internal agents, culminating in a financial strategy agent. This phase utilizes **LangChain**, **LangGraph**, and **FastMCP** to integrate external data tools and technical indicators (SMA, RSI, MACD).

### Phase 2: Automated Evaluation (DeepEval)
We set up and validated **DeepEval** as a reliable LLM-as-a-judge framework by running it against our financial agent. We verified the consistency of our **GPT-4.1** judge across multiple iterations, testing against multilingual inputs, typos, and edge-case scenarios.

### Phase 3: Benchmarking & Hallucination Audit
We designed an original evaluation methodology to audit 15 external endpoints provided by Michelin, split across two stacks: 6 ADK fact-checking agents and 9 OpenClaw generalist agents. We evaluated them across 5 hallucination axes:

* **Factual:** Invention of data not present in sources.
* **Rare Facts:** Failure to source highly specialized or obscure information.
* **Logic:** Contradictory reasoning or failure to detect logical fallacies.
* **Temporal:** Erroneous extrapolations of events occurring after the model's knowledge cutoff (2024-2026).
* **Stereotypes:** Biased or discriminatory responses on sensitive topics.
