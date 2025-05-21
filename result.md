FAISS Search Performance Summary
This summary presents the results of FAISS search operations using two different embedding dimensions (1536 and 3072) and two similarity metrics (cosine and L2).

Across all tests, a total of 80,796 queries were processed.

Key Observations:
Embedding Dimension 1536:
Cosine Similarity:
Top-1 Accuracy: 0.6691
Time Taken: 31.29 seconds
RAM Increase: 1114.94 MB

L2 Similarity:
Top-1 Accuracy: 0.6691
Time Taken: 33.07 seconds
RAM Increase: 1039.78 MB

Embedding Dimension 3072:
Cosine Similarity:
Top-1 Accuracy: 0.6900
Time Taken: 62.27 seconds
RAM Increase: 1663.41 MB

L2 Similarity:
Top-1 Accuracy: 0.6900
Time Taken: 62.19 seconds
RAM Increase: 2063.27 MB
Summary of Performance:

Embedding Dimension	Similarity Metric	Top-1 Accuracy	Time Taken (seconds)	RAM Increase (MB)
1536	Cosine	0.6691	31.29	1114.94
1536	L2	0.6691	33.07	1039.78
3072	Cosine	0.6900	62.27	1663.41
3072	L2	0.6900	62.19	2063.27
