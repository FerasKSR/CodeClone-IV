## Code Search Task
The code search task is to find the source code file most related to a given code snippet.

The code search task involves code embedding and vector search. Every source code file and search queries (code snippets) are encoded into vectors. The goal is to find the file whose vector is closest to that of the search query.

In this task, you are asked to provide a report to management on the code search project. The report should include information on **cost, accuracy, speed, and memory** for the vector search component, while the code embeddings will be provided as part of the dataset.

## Dataset Description:
List of source code files where each file has:
- A file ID
- Two sets of embeddings:
  - Embedding 1 with a dimensionality of 1536
  - Embedding 2 with a dimensionality of 3072
- List of search queries is provided. Each query includes:
  - The encoded representation of the code snippet (using both embeddings)
  - The file ID of the original file from which the snippet was extracted


You can find the dataset [here](https://drive.google.com/file/d/1s5VOQdbQcavqJX_7GKjOsyb6Ab5vPYQg/view?usp=share_link)
 
## Instructions
- You should use [Faiss](https://github.com/facebookresearch/faiss) to find the matching results between the original file and the code snippet.
  - *Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search.*
- You have to push the source code to this repository.
- It is up to you to show and discuss the results regarding the cost, accuracy, speed, and memory.
  - For cost and memory, we don't expect specific numbers
- Which machine would you recommend from the AWS G4dn family [[link](https://instances.vantage.sh/aws/ec2/g4dn.xlarge)]?
- How much data can we support?
- Additional considerations:
  - Compare between L2 distance vs. cosine similarity
  - Explore and implement quantization approaches
- Extra Step
  - Apply PCA to reduce embedding dimensionality

### Note

Feel free to make reasonable assumptions (or ask us), and add/remove any part of the report as you see fit.
Also, note that we would rather you present an indepth analysis on few aspects instead of a shallow analysis on many aspects.
We expect you to be pragmatic and decide what to focus on, yourself.