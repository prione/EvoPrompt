# LLM Prompt Optimization with Evolutionary Algorithms

This is a framework for automatically adjusting LLM prompts using evolutionary algorithms. It is based on the following paper:
[Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/abs/2309.08532)

## Compatibility
This framework runs on OpenAI-compatible servers.

## Require
   ```sh
   pip install faiss-cpu sentense-transformers numpy openai
   ```

## Usage
1. Modify `config.py` to fit your environment.
2. Run the script:
   ```sh
   python main.py
   ```

## Evaluation
The evaluation metric for prompts is only cosine similarity, so please adjust it according to the task.
