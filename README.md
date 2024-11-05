# [NeurIPS 2024] Self-Retrieval: End-to-End Information Retrieval with One Large Language Model
## What is Self-Retrieval?

**Self-Retrieval** is an end-to-end LLM-driven information retrieval architecture that unifies indexing, retrieval, and reranking in a single LLM:
- Indexing: Self-Retrieval internalizes the entire corpus into its parameters through self-supervised learning, enabling the model to process passages internally without relying on external indices.
- Retrieval: Given an input query $q$, Self-Retrieval generates relevant passage $p$ using the knowledge embedded within its parameters, which is different from dense retrieval or generative retrieval that rely on embedding or document identifiers as proxies of passage.
- Reranking: After generating passage $p$, Self-Retrieval assesses its relevance to the query $q$ through self-assessment. The output logits provide the basis for reranking candidate passages.
Experimental results demonstrate that Self-Retrieval not only outperforms existing retrieval approaches by a significant margin, but also substantially enhances the performance of LLM-driven downstream applications like retrieval-augmented generation. 
For more information, checkout our [publications](https://arxiv.org/pdf/2403.00801).
