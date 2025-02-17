# Finance-LLM

In this notebook we will:
1. Fine tune a local model of LLaMa 3 on form 10-K Contextual Q&A Data using supervised fine tuning & Low Rank Adaptation
2. Set up an SEC Data pipeline to retrieve recent form 10-K's
3. Use local embedding and in memory vector stores to create a retrieval function
4. Combine everything above to make a simple financial analysis RAG agent!

HuggingFace Token Found: https://huggingface.co/settings/tokens  
Free SEC API Key Here: https://sec-api.io/

---
## **Part 1: Fine Tuning LLaMa 3 with Unsloth**

We will be using the built in GPU on Colab to do all our fine tuning needs, using the [Unsloth Library](https://github.com/unslothai/unsloth).

Much of the below code is augmented from [Unsloth Documentation!](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=AEEcJ4qfC7Lp)

We will be using Meta's [LLaMa 3 8b Instruct Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).  
 **NOTE**: This is a gated model, you must request access on HF and pass in your HF token in the below step.

Parameters used while applying LoRA (Low-Rank Adaptation):
1. **r**: The rank of the low-rank adaptation matrix. This determines the capacity of the adapter to capture additional information. Higher ranks allow capturing more complex patterns but also increase computational overhead.

2. **target_modules**: List of module names within the model to which the LoRA adapters should be applied. These modules typically include the projections within transformer layers (e.g., query, key, value projections) and other key transformation points.
  - **q_proj**: Projects input features to query vectors for attention mechanisms.
  - **k_proj**: Projects input features to key vectors for attention mechanisms.
  - **v_proj**: Projects input features to value vectors for attention mechanisms.
  - **o_proj**: Projects the output of the attention mechanism to the next layer.
  - **gate_proj**: Applies gating mechanisms to modulate the flow of information.
  - **up_proj**: Projects features to a higher dimensional space, often used in feed-forward networks.
  - **down_proj**: Projects features to a lower dimensional space, often used in feed-forward networks.

These layers are typically involved in transformer-based models, facilitating various projections and transformations necessary for the attention mechanism and feed-forward processes.

3. **lora_alpha**: A scaling factor for the LoRA adapter. It controls the impact of the adapter on the model's outputs. Typically set to a small positive integer.

4. **lora_dropout**: Dropout rate applied to the LoRA adapters. Dropout helps in regularizing the model, but setting it to 0 means no dropout, which is often optimal for performance.

5. **bias**: This specifies how biases should be handled in the LoRA adapters. Setting it to "none" indicates no bias is used, which is optimized for performance, although other options are available depending on the use case.

6. **use_gradient_checkpointing**: Enables gradient checkpointing, which helps to save memory during training by not storing all intermediate activations. "unsloth" is a setting optimized for very long contexts, but it can also be set to True.

7. **random_state**: A seed for the random number generator to ensure reproducibility. This makes sure that the results are consistent across different runs of the code.

### **Preparing the Fine Tuning Dataset**

We will be using a HF dataset of Financial Q&A over form 10ks, provided by user [Virat Singh](https://github.com/virattt) here https://huggingface.co/datasets/virattt/llama-3-8b-financialQA

The following code below formats the entries into the prompt defined first for training, being careful to add in special tokens. In this case our End of Sentence token is <|eot_id|>. More LLaMa 3 special tokens [here](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

### **Defining the Trainer Arguments**

We will be setting up and using HuggingFace Transformer Reinforcement Learning (TRL)'s [Supervised Fine-Tuning Trainer](https://huggingface.co/docs/trl/sft_trainer)

**Supervised fine-tuning** is a process in machine learning where a pre-trained model is further trained on a specific dataset with labeled examples. During this process, the model learns to make predictions or classifications based on the labeled data, improving its performance on the specific task at hand. This technique leverages the general knowledge the model has already acquired during its initial training phase and adapts it to perform well on a more targeted set of examples. Supervised fine-tuning is commonly used to customize models for specific applications, such as sentiment analysis, object recognition, or language translation, by using task-specific annotated data.

1. **model**: The model to be fine-tuned. This is the pre-trained model that will be adapted to the specific training data.

2. **tokenizer**: The tokenizer associated with the model. It converts text data into tokens that the model can process.

3. **train_dataset**: The dataset used for training. This is the collection of labeled examples that the model will learn from during the fine-tuning process.

4. **dataset_text_field**: The field in the dataset containing the text data. This specifies which part of the dataset contains the text that the model will be trained on.

5. **max_seq_length**: The maximum sequence length for the training data. This limits the number of tokens per input sequence to ensure they fit within the model's processing capacity.

6. **dataset_num_proc**: The number of processes to use for data loading. This can speed up data loading by parallelizing it across multiple processes.

7. **packing**: A boolean indicating whether to use sequence packing. Sequence packing can speed up training by combining multiple short sequences into a single batch.

8. **args**: A set of training arguments that configure the training process. These include various hyperparameters and settings:

    - **per_device_train_batch_size**: The batch size per device during training. This determines how many examples are processed together in one forward/backward pass.
    
    - **gradient_accumulation_steps**: The number of gradient accumulation steps to perform before updating the model parameters. This allows for effectively larger batch sizes without requiring more memory.
    
    - **warmup_steps**: The number of warmup steps for the learning rate scheduler. During these steps, the learning rate increases gradually to the initial value.
    
    - **max_steps**: The total number of training steps. This defines how many batches of training data the model will be trained on.
    
    - **num_train_epochs**: The number of training epochs (uncommented in the example). This defines how many times the entire training dataset will be passed through the model.
    
    - **learning_rate**: The learning rate for the optimizer. This controls how much to adjust the model's weights with respect to the gradient during training.
    
    - **fp16**: A boolean indicating whether to use 16-bit floating point precision for training if bfloat16 is not supported. This can speed up training and reduce memory usage.
    
    - **bf16**: A boolean indicating whether to use bfloat16 precision for training if supported. This can also speed up training and reduce memory usage while maintaining higher precision than fp16.
    
    - **logging_steps**: The number of steps between logging events. This controls how frequently training progress and metrics are logged.
    
    - **optim**: The optimizer to use. In this case, AdamW with 8-bit precision, which can improve efficiency for large models.
    
    - **weight_decay**: The weight decay to apply to the model parameters. This is a regularization technique to prevent overfitting by penalizing large weights.
    
    - **lr_scheduler_type**: The type of learning rate scheduler to use. This controls how the learning rate changes over time during training.
    
    - **seed**: The seed for random number generation. This ensures reproducibility of results by controlling the randomness in training.
    
    - **output_dir**: The directory to save the output models and logs. This specifies where the trained model and training logs will be stored.


---
# **Part 2: Setting Up SEC 10-K Data Pipeline & Retrieval Functionality**

Now that we have our fine tuned language model, inference functions, and a desired prompt format, we need to now set up the RAG pipeline to inject the relevant context into each generation.

The flow will follow as such:

*User Question* -> *Context Retrieval from 10-K* -> *LLM Answers User Question Using Context*

To do this we will need to be able to:
1. Gather specific from 10-K's
2. Parse and chunk the text in them
3. Vectorize and embed the chunks into a vector Database
4. Set up a retriever to semantically search the user's questions over the database to return relevant context

A **Form 10-K** is an annual report required by the U.S. Securities and Exchange Commission, that gives a comprehensive summary of a company's financial performance.

### **Function For 10-K Retrieval**

To do this easier, we're taking advantage of the SEC API https://sec-api.io/. It is free to sign up, and you get 100 API calls a day to use, each time we load a ticker's symbol it will use 3 calls.

For this project, we'll be focused on loading only sections **1A** and **7**
- **1A**: Risk Factors
- **7**: Management's Discussion and Analysis of Financial Condition and Results of Operations

### **Setting Up Embeddings Locally**

In the spirit of local and fine tuned models, we'll be using an open source embedding model, [Beijing Academy of Artificial Intelligence's - Large English Embedding Model](https://huggingface.co/BAAI/bge-large-en-v1.5). More details on their open source model available in their [GitHub repo](https://github.com/FlagOpen/FlagEmbedding)!

**Embeddings** are numerical representations of data, typically used to convert complex, high-dimensional data into a lower-dimensional space where similar data points are closer together. In the context of natural language processing (NLP), embeddings are used to represent words, phrases, or sentences as vectors of real numbers. These vectors capture semantic relationships, meaning that words with similar meanings are represented by vectors that are close together in the embedding space.

**Embedding models** are machine learning models that are trained to create these numerical representations. They learn to encode various types of data into embeddings that capture the essential characteristics and relationships within the data. For example, in NLP, embedding models like Word2Vec, GloVe, and BERT are trained on large text corpora to produce word embeddings. These embeddings can then be used for various downstream tasks, such as text classification, sentiment analysis, or machine translation. In this case we'll be using it for semantic similarity

### **Processing & Defining the Vector Database**

In this flow we get the data from the above defined SEC API functions, and then go through Three steps:
1. Text Splitting
2. Vectorizing
3. Retrieval Function Setup

**Text splitting** is the process of breaking down large documents or text data into smaller, manageable chunks. This is often necessary when dealing with extensive text data, such as legal documents, financial reports, or any lengthy articles. The purpose of text splitting is to ensure that the data can be effectively processed, analyzed, and indexed by machine learning models and databases.

**Vector databases** store data in the form of vectors, which are numerical representations of text, images, or other types of data. These vectors capture the semantic meaning of the data, allowing for efficient similarity search and retrieval.

The Vector DB we're using here is the [Facebook AI Semantic Search](https://ai.meta.com/tools/faiss/) library, a lightweight an in memory (don't need to save this to a disk) solution that is not as powerful as other Vector DB's but will work great for this use case

**How They Use Split Documents and Embeddings Together:**
1. **Embeddings:** When text data is split into smaller chunks, each chunk is converted into a numerical vector (embedding) using an embedding model. These embeddings capture the semantic relationships and meaning of the text.
2. **Storage:** The vector database stores these embeddings along with references to the original text chunks.
3. **Indexing:** The database indexes the vectors to allow for fast and efficient similarity searches. This indexing process organizes the vectors in a way that makes it easy to find similar vectors quickly.
4. **Usage:** When a query is made, the vector database searches for the most similar vectors (text chunks) to the query vector, retrieving the relevant text chunks based on their semantic similarity.


### **Retrieval**

**Description:**
Retrieval is the process of querying a vector database to find and return relevant text chunks or documents that match a given query. This involves searching through the indexed embeddings to identify the ones that are most similar to the query.

**How It Works:**
1. **Query Embedding:** When a query is made, it is first converted into an embedding using the same embedding model used for the text chunks.
2. **Similarity Search:** The retriever searches the vector database for embeddings that are similar to the query embedding. This similarity is often measured using distance metrics like cosine similarity or Euclidean distance.
3. **Document Retrieval:** The retriever then retrieves the original text chunks or documents associated with the similar embeddings.
4. **Context Assembly:** The retrieved text chunks are assembled to provide a coherent context or answer to the query.

In this function, the query is used to invoke the retriever, which returns a list of documents. The content of these documents is then extracted and returned as the context for the query.
