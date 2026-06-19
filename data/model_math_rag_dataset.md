# Mathematical Foundations for Local LLM and RAG Systems

## 1. Overview

This document is a compact technical primer on the mathematical ideas behind modern language-model applications. It is intended to be used as a small retrieval dataset for a local Retrieval-Augmented Generation (RAG) prototype. The focus is not on training a large language model from scratch, but on the core concepts that help an engineer understand how text can be represented, compared, optimized, and retrieved.

A language-model application usually combines several layers. First, text is represented numerically so that software can process it. Second, a model transforms these numerical representations through many learned operations. Third, during training, the model is adjusted by minimizing a loss function. Fourth, during inference, the model uses its learned parameters to generate or rank outputs. In RAG systems, a separate retrieval layer searches for relevant context and injects it into the prompt before generation.

The most important mathematical themes are vectors, matrices, similarity, loss functions, gradients, and optimization. Vectors allow text, tokens, or documents to be represented as points in a high-dimensional space. Matrices represent learned transformations that map one vector representation into another. Similarity functions measure how close two representations are. Loss functions define what the model is trying to improve. Gradients indicate how each parameter should change to reduce the loss. Optimization algorithms use those gradients to update model parameters over many training steps.

This document is organized as a short reference. Each section is written as a self-contained chunk-friendly explanation so that a RAG system can retrieve the most relevant passages for technical questions.

---

## 2. Representing Information as Vectors

A vector is an ordered list of numbers. In machine learning, vectors are used to represent data in a form that a model can process. If a vector has `n` real-valued components, it is commonly written as an element of `R^n`. For example, a vector in `R^4` might contain four numeric features that describe an object.

In natural language processing, raw text cannot be processed directly by mathematical models. Text must first be converted into numbers. A simple representation might count how many times each word appears in a document. A more advanced representation, called an embedding, maps words, sentences, or documents into dense vectors. A dense vector usually contains many real numbers, and each number contributes to the meaning of the representation.

The key idea behind vector representations is that semantic relationships can be reflected as geometric relationships. If two pieces of text have similar meanings, their vectors should be close to each other in the embedding space. If they are unrelated, their vectors should be farther apart. This is why vector search is useful in RAG systems: a user query and document chunks can be embedded into the same space, and the system can retrieve chunks whose vectors are close to the query vector.

For example, a sentence about "model deployment" should be closer to a document chunk about "serving an LLM through an HTTP endpoint" than to a chunk about "gradient descent during training." The retrieval layer does not understand text like a human. It compares numerical vectors produced by an embedding model.

---

## 3. Embeddings and Semantic Space

An embedding is a learned numerical representation of an item such as a word, sentence, paragraph, image, or document. In text systems, embedding models are trained so that semantically related text receives similar vectors. The output dimension might be 384, 768, 1024, or another fixed size, depending on the embedding model.

Embeddings are useful because they compress meaning into a form that can be searched efficiently. Instead of comparing text using exact keywords only, an embedding model can capture softer relationships. For example, "authorization," "access control," and "permissions" may be close in embedding space even if the exact words differ. This is important for technical documentation search, because users rarely ask questions using exactly the same wording as the document.

A RAG system usually embeds each document chunk once during indexing. These vectors are stored in a vector index such as FAISS or ChromaDB. At query time, the user question is embedded using the same embedding model. The query vector is compared against the stored document vectors, and the system retrieves the most similar chunks.

The embedding model used for indexing and the embedding model used for querying must be the same or compatible. If document chunks are embedded with one model and queries are embedded with a different model, the vectors may not live in the same semantic space, and retrieval quality can become unreliable.

---

## 4. Measuring Similarity Between Vectors

Once text has been converted into vectors, the system needs a way to measure similarity. A common method is cosine similarity. Cosine similarity compares the angle between two vectors rather than their raw length. If two vectors point in a similar direction, their cosine similarity is high. If they point in unrelated directions, their similarity is lower.

For two vectors `a` and `b`, cosine similarity is defined as:

`cosine_similarity(a, b) = (a · b) / (||a|| ||b||)`

Here, `a · b` is the dot product, and `||a||` and `||b||` are the vector norms. The result is usually between -1 and 1, although in many embedding systems most useful similarities are non-negative. A higher value indicates greater similarity.

Vector databases and indexes often use cosine similarity, inner product, or Euclidean distance. These are different scoring methods, but they serve the same general purpose: ranking stored vectors according to their closeness to the query vector. In practice, many systems normalize vectors and use inner product search as an efficient approximation of cosine similarity.

In a RAG pipeline, the retrieval result is usually a ranked list. The top-ranked chunks are considered the most relevant. The assignment for this prototype asks for retrieving the top 3 relevant chunks. That means the system should return the three chunks whose vectors are most similar to the query vector according to the selected similarity metric.

---

## 5. Matrices as Learned Transformations

A matrix is a rectangular table of numbers. In neural networks, matrices are used to transform vectors from one representation into another. A typical linear transformation has the form:

`z = Wx + b`

Here, `x` is the input vector, `W` is a weight matrix, `b` is a bias vector, and `z` is the output vector before activation. The weight matrix contains learned parameters. During training, the model adjusts these parameters so that the transformation becomes useful for the task.

If the input vector has dimension 384 and the output vector has dimension 128, then the matrix `W` maps from a 384-dimensional space to a 128-dimensional space. This transformation can rotate, scale, combine, and project features. Neural networks apply many such transformations in sequence.

A single linear transformation is limited. If a network only stacks linear transformations without nonlinear activations, the entire network is still equivalent to one larger linear transformation. Nonlinear activation functions allow the model to represent more complex patterns. This is one reason modern neural networks can model complicated relationships in text and other data.

In large language models, the actual architecture is more complex than a simple feed-forward network. Transformers use attention mechanisms, layer normalization, feed-forward blocks, and residual connections. Even so, matrices remain central. Most learned parts of the model are represented by large matrices of parameters.

---

## 6. Neural Networks and Activation Functions

A neural network is a composition of mathematical functions. Each layer receives an input representation and produces a new representation. A simple layer applies a linear transformation and then a nonlinear activation:

`h = activation(Wx + b)`

The activation function introduces nonlinearity. Common activation functions include ReLU, GELU, sigmoid, and tanh. ReLU is defined as `max(0, x)`. GELU is widely used in transformer-based models because it provides a smooth nonlinear transformation.

Nonlinearity is essential because real-world patterns are rarely linear. For example, the meaning of a sentence is not just the sum of independent word meanings. Context, order, and interaction matter. Nonlinear transformations allow a model to learn complex combinations of features.

In transformer language models, layers repeatedly transform token representations. At the beginning, tokens are represented as embeddings. After many layers, each token representation contains contextual information from other tokens. The final representation is used to predict the next token or to produce another task-specific output.

A RAG application usually does not train the LLM. Instead, it uses a pretrained model at inference time. However, understanding neural network layers helps explain why model behavior depends on learned representations and why prompt context can influence the generated answer.

---

## 7. Loss Functions

A loss function measures how wrong a model is on a training example or batch. Training tries to reduce this loss. The choice of loss function depends on the task. For classification, cross-entropy loss is common. For regression, mean squared error is common. For language modeling, the model is often trained to predict the next token, and cross-entropy compares the predicted token distribution with the correct next token.

The model produces scores or probabilities for possible outputs. The loss function converts the difference between prediction and target into a single number. A lower loss usually means the model is doing better on the training objective.

For example, in next-token prediction, the model receives a sequence of tokens and tries to predict the next token at each position. If the correct token receives high probability, the loss is low. If the correct token receives low probability, the loss is high. Over many examples, training adjusts the model so that correct tokens become more likely.

It is important to remember that minimizing training loss does not automatically guarantee useful real-world behavior. The model may overfit, fail on unseen data, or produce fluent but incorrect answers. In practical LLM systems, evaluation, guardrails, retrieval quality, and monitoring are all important in addition to the training objective.

---

## 8. Gradients

A gradient describes how a loss function changes when model parameters change. If a model has millions or billions of parameters, the gradient contains information about how each parameter should be adjusted to reduce the loss.

The gradient of a function points in the direction of steepest increase. To minimize a loss, optimization algorithms usually move parameters in the opposite direction of the gradient. This idea is the basis of gradient descent.

For a parameter `w` and loss `L`, a simple update rule is:

`w_new = w_old - learning_rate * gradient`

The learning rate controls the step size. If the learning rate is too high, training may become unstable and overshoot good solutions. If it is too low, training may be very slow. Choosing and scheduling the learning rate is a major part of optimization.

Gradients are calculated using backpropagation. Backpropagation applies the chain rule from calculus to compute how the final loss depends on each intermediate computation and parameter. Modern machine learning frameworks perform this automatically through automatic differentiation.

---

## 9. Optimization

Optimization is the process of updating model parameters to reduce the loss function. The simplest method is gradient descent, where parameters are updated using gradients computed over the entire training dataset. In practice, full-batch gradient descent is often too expensive, so training usually uses mini-batches. A mini-batch is a small subset of training examples processed at each step.

Stochastic Gradient Descent (SGD) updates parameters using gradients from one example or a mini-batch. More advanced optimizers, such as Adam and AdamW, maintain additional moving averages of gradients and squared gradients. These methods often train deep networks more efficiently than plain SGD.

Optimization involves trade-offs. A model should reduce training loss, but it should also generalize to new data. Regularization, validation sets, early stopping, and careful data design help reduce overfitting. In large language models, optimization also involves huge compute resources, distributed training, and careful numerical stability.

For a RAG prototype, optimization usually refers less to training the LLM and more to optimizing retrieval and inference behavior. Examples include choosing chunk size, selecting a similarity metric, limiting context length, reducing latency, and improving prompt structure.

---

## 10. Backpropagation at a High Level

Backpropagation is the algorithm used to compute gradients in neural networks. It has two main phases: a forward pass and a backward pass. During the forward pass, the input moves through the network and produces a prediction. The loss function then compares the prediction with the target. During the backward pass, the system computes how much each parameter contributed to the loss.

Backpropagation relies on the chain rule. If a model is a composition of functions, the chain rule allows the derivative of the final output with respect to earlier parameters to be computed step by step. This makes training deep networks possible.

In practice, engineers rarely implement backpropagation manually for standard models. Frameworks such as PyTorch and TensorFlow track operations and compute gradients automatically. However, understanding the idea is valuable because it explains how models learn from errors.

In local LLM applications, backpropagation is usually not part of the runtime system. A deployed LLM performs inference, not training. The model's parameters are loaded and used to generate outputs, but they are not updated during normal chat interactions. RAG improves answers by adding relevant context, not by changing the model weights.

---

## 11. Training, Fine-Tuning, and Inference

Training, fine-tuning, and inference are different stages. Training a model from scratch means learning parameters from a large dataset, often requiring substantial compute. Fine-tuning starts from a pretrained model and adjusts it on a smaller, task-specific dataset. Inference means using a trained model to generate outputs without changing the model parameters.

Most application teams do not train large language models from scratch. They use pretrained models through APIs or local inference engines. If needed, they may fine-tune smaller models or use prompt engineering and RAG to adapt behavior.

RAG is often chosen because it allows the system to use domain-specific knowledge without retraining the model. Instead of modifying weights, the system retrieves relevant documents and includes them in the prompt. This makes the approach easier to update: changing the knowledge base does not require retraining the model.

Inference systems still require engineering decisions. The team must choose a model, an inference engine, quantization level, context window, timeout policy, streaming behavior, and error handling. These decisions affect cost, latency, reliability, and answer quality.

---

## 12. Quantization and Model Size

Quantization reduces the numerical precision used to store and compute model parameters. For example, instead of storing weights as 16-bit floating point numbers, a quantized model may store them using 8-bit or 4-bit representations. Lower precision can reduce memory usage and improve speed, especially on limited hardware.

The trade-off is that aggressive quantization may reduce output quality. A 4-bit model may be much smaller and faster than a full-precision model, but it may also produce less accurate or less stable answers. The best quantization level depends on the use case, hardware, latency requirements, and acceptable quality.

Local inference engines often support quantized models because developers may need to run LLMs on consumer hardware or edge devices. A small instruction-tuned model with quantization can be sufficient for a prototype, testing environment, or constrained deployment.

For an assessment project, quantization can be evaluated by measuring tokens per second and peak RAM or VRAM usage. The report should also include a qualitative comparison of answer quality. Speed and memory numbers alone are not enough; a very fast model is not useful if its answers are consistently poor.

---

## 13. RAG Pipeline Mechanics

A Retrieval-Augmented Generation pipeline combines retrieval with generation. The retrieval part searches for relevant information. The generation part uses the retrieved context to produce an answer. The purpose is to improve factual grounding and allow the system to answer questions about a specific dataset.

A typical RAG indexing flow includes the following steps:

1. Load documents from files.
2. Split documents into chunks.
3. Embed each chunk using an embedding model.
4. Store the vectors and metadata in a vector index.

A typical RAG query flow includes these steps:

1. Receive the user query.
2. Embed the query using the same embedding model.
3. Search the vector index for the top-k similar chunks.
4. Build an augmented prompt containing the retrieved context.
5. Send the prompt to the LLM.
6. Return the answer, optionally with citations or trace information.

Chunk size affects retrieval quality. If chunks are too small, they may lack enough context. If chunks are too large, they may include irrelevant information and reduce precision. Overlap helps preserve continuity between adjacent chunks. A simple prototype can use fixed-size chunks with overlap, while a more advanced system can split by headings, paragraphs, or semantic boundaries.

---

## 14. Vector Stores and In-Memory Indexes

A vector store keeps embeddings and allows similarity search. In a lightweight prototype, the vector store can be in memory. This means the index is built when the application starts and is not necessarily persisted as a production database. FAISS is a common library for efficient similarity search over dense vectors. ChromaDB is another popular option that provides a higher-level interface and optional persistence.

An in-memory index is simple and fast for small datasets. It is a good fit when the document set has only a few pages and can be loaded at startup. The downside is that data must be re-indexed when the process restarts unless persistence is added. For a small assessment project, this is acceptable and matches the requirement for a lightweight RAG system.

For larger production systems, teams often use a dedicated vector database such as Qdrant, Weaviate, Milvus, or a managed cloud service. These systems provide persistence, filtering, scaling, replication, and operational tooling. However, they are unnecessary for a small local prototype.

A good RAG implementation should return not only chunk text but also metadata such as source file, chunk ID, and score. This makes debugging easier and allows logs to show why a particular answer was generated.

---

## 15. Evaluation and Debugging of Retrieval

Retrieval quality should be tested directly. A simple test log should include the user query, the top retrieved chunks, and similarity scores. This helps verify that the system retrieves the right material before the LLM is involved.

For example, a query about "why gradients are used in training" should retrieve chunks about gradients, loss functions, and optimization. A query about "how RAG finds relevant context" should retrieve chunks about embeddings, vector stores, and query flow. If the retrieved chunks are unrelated, the problem is likely in chunking, embeddings, normalization, or the similarity search configuration.

Debugging RAG is easier when generation is temporarily disabled. If the retrieval step is wrong, the final answer will often be wrong even if the LLM is strong. Therefore, a robust implementation should expose the retrieval trace and allow the developer to inspect the selected chunks.

Evaluation can be improved with a small set of expected questions. For each question, the developer can identify which document sections should be retrieved. This does not require a complex benchmark. Even five well-designed test questions can reveal many retrieval problems.

---

## 16. Relationship Between RAG and Agentic Systems

RAG can be used as a tool inside an agentic system. An agent receives a user prompt and decides what action to take. If the prompt requires information from the knowledge base, the agent can call a retrieval tool. If the prompt can be answered directly, the agent may respond without retrieval.

In a simple implementation, the RAG pipeline can be wrapped as a Python function such as `search_knowledge_base(query: str) -> list[RetrievedChunk]`. The agent can call this function and then use the returned chunks to build a grounded answer. This design keeps the RAG logic modular and testable.

The important separation is that the retrieval pipeline should not depend on the agent. The RAG system should be usable independently: given a query, it returns the top-k relevant chunks. Later, the agent layer can decide when to call it. This separation makes the application easier to test and maintain.

In production, agentic systems require careful tracing. The system should log when a tool was called, what input was passed to it, what results were returned, and how the final answer was produced. This is especially important for debugging, observability, and safety.

---

## 17. Practical Engineering Considerations

A local RAG prototype should be small, reproducible, and easy to run. The dataset should be committed to the repository. The embedding model name should be configured in one place. The index should be built deterministically from the dataset. The test script should print enough information for a reviewer to understand the retrieval result.

The implementation should avoid unnecessary complexity. For example, it is not necessary to deploy a standalone vector database if the assignment asks for an in-memory store. It is also not necessary to implement a web API before the retrieval pipeline is proven.

Good engineering practice includes clear naming, small functions, and explicit data structures. A retrieved chunk can be represented with fields such as `chunk_id`, `text`, `source`, and `score`. This makes the retrieval output easy to log and later easy to pass into an agent or API.

The RAG layer should be designed so it can later be reused by the agent and FastAPI layers. This usually means implementing it as a class or module rather than only as a script. The script should be a thin executable wrapper around reusable application code.

---

## 18. Summary

The mathematical foundation of LLM applications begins with representing information as vectors. Embeddings map text into semantic vector spaces. Similarity search retrieves nearby vectors. Neural networks use matrices and nonlinear functions to transform representations. Training uses loss functions, gradients, backpropagation, and optimization to adjust parameters. Inference uses the trained parameters to generate outputs.

RAG systems do not usually change model weights. Instead, they improve answer relevance by retrieving useful context and placing it into the prompt. This makes RAG especially practical for domain-specific assistants, technical documentation search, and systems where knowledge must be updated without retraining.

A strong local RAG prototype should demonstrate the full retrieval flow: load a dataset, split it into chunks, embed chunks, store vectors in an in-memory index, embed a user query, retrieve the top 3 relevant chunks, and log the retrieval trace. This provides a clean foundation for a later agentic orchestrator and streaming API.
