 Memorization and Knowledge Injection in Gated LLMs       

1.  [1 Introduction](https://arxiv.org/html/2504.21239v1#S1 "In Memorization and Knowledge Injection in Gated LLMs")
2.  [2 Related Works](https://arxiv.org/html/2504.21239v1#S2 "In Memorization and Knowledge Injection in Gated LLMs")
    1.  [2.1 Continual Learning and Catastrophic Forgetting](https://arxiv.org/html/2504.21239v1#S2.SS1 "In 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")
    2.  [2.2 Knowledge Injection](https://arxiv.org/html/2504.21239v1#S2.SS2 "In 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")
    3.  [2.3 Gating Networks](https://arxiv.org/html/2504.21239v1#S2.SS3 "In 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")
3.  [3 Methods](https://arxiv.org/html/2504.21239v1#S3 "In Memorization and Knowledge Injection in Gated LLMs")
    1.  [3.1 Problem Formulation](https://arxiv.org/html/2504.21239v1#S3.SS1 "In 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
    2.  [3.2 Model and Datasets](https://arxiv.org/html/2504.21239v1#S3.SS2 "In 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
    3.  [3.3 Memory Embedded in Gated LLMs (MEGa)](https://arxiv.org/html/2504.21239v1#S3.SS3 "In 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
        1.  [3.3.1 Fine-tuning](https://arxiv.org/html/2504.21239v1#S3.SS3.SSS1 "In 3.3 Memory Embedded in Gated LLMs (MEGa) ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
        2.  [3.3.2 Inference](https://arxiv.org/html/2504.21239v1#S3.SS3.SSS2 "In 3.3 Memory Embedded in Gated LLMs (MEGa) ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
    4.  [3.4 Evaluation](https://arxiv.org/html/2504.21239v1#S3.SS4 "In 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
        1.  [3.4.1 Memory Recall](https://arxiv.org/html/2504.21239v1#S3.SS4.SSS1 "In 3.4 Evaluation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
        2.  [3.4.2 QA](https://arxiv.org/html/2504.21239v1#S3.SS4.SSS2 "In 3.4 Evaluation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
        3.  [3.4.3 General Knowledge Baseline](https://arxiv.org/html/2504.21239v1#S3.SS4.SSS3 "In 3.4 Evaluation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
    5.  [3.5 Continual Learning Baselines](https://arxiv.org/html/2504.21239v1#S3.SS5 "In 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")
4.  [4 Results](https://arxiv.org/html/2504.21239v1#S4 "In Memorization and Knowledge Injection in Gated LLMs")
    1.  [4.1 Memory Recall](https://arxiv.org/html/2504.21239v1#S4.SS1 "In 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")
    2.  [4.2 Question-Answering](https://arxiv.org/html/2504.21239v1#S4.SS2 "In 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")
    3.  [4.3 Internal RAG](https://arxiv.org/html/2504.21239v1#S4.SS3 "In 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")
    4.  [4.4 General Knowledge Retention](https://arxiv.org/html/2504.21239v1#S4.SS4 "In 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")
    5.  [4.5 Composite Knowledge](https://arxiv.org/html/2504.21239v1#S4.SS5 "In 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")
5.  [5 Discussion](https://arxiv.org/html/2504.21239v1#S5 "In Memorization and Knowledge Injection in Gated LLMs")
6.  [A Code availability](https://arxiv.org/html/2504.21239v1#A1 "In Memorization and Knowledge Injection in Gated LLMs")
7.  [B Supervised Fine-Tuning](https://arxiv.org/html/2504.21239v1#A2 "In Memorization and Knowledge Injection in Gated LLMs")
8.  [C Supplementary Tables and Figures](https://arxiv.org/html/2504.21239v1#A3 "In Memorization and Knowledge Injection in Gated LLMs")
9.  [D Fine-Tuning Settings](https://arxiv.org/html/2504.21239v1#A4 "In Memorization and Knowledge Injection in Gated LLMs")
10.  [E Prompts](https://arxiv.org/html/2504.21239v1#A5 "In Memorization and Knowledge Injection in Gated LLMs")
    1.  [E.1 Dataset Generation](https://arxiv.org/html/2504.21239v1#A5.SS1 "In Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        1.  [E.1.1 Fictional Character Dataset Stories](https://arxiv.org/html/2504.21239v1#A5.SS1.SSS1 "In E.1 Dataset Generation ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        2.  [E.1.2 Generating Paraphrases](https://arxiv.org/html/2504.21239v1#A5.SS1.SSS2 "In E.1 Dataset Generation ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        3.  [E.1.3 Generating QA](https://arxiv.org/html/2504.21239v1#A5.SS1.SSS3 "In E.1 Dataset Generation ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
    2.  [E.2 Compositional Question Dataset Construction](https://arxiv.org/html/2504.21239v1#A5.SS2 "In Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
    3.  [E.3 GPT judge](https://arxiv.org/html/2504.21239v1#A5.SS3 "In Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
    4.  [E.4 MMLU Prompt](https://arxiv.org/html/2504.21239v1#A5.SS4 "In Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
    5.  [E.5 Evaluation Prompts](https://arxiv.org/html/2504.21239v1#A5.SS5 "In Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        1.  [E.5.1 Recall Prompt](https://arxiv.org/html/2504.21239v1#A5.SS5.SSS1 "In E.5 Evaluation Prompts ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        2.  [E.5.2 QA Prompt](https://arxiv.org/html/2504.21239v1#A5.SS5.SSS2 "In E.5 Evaluation Prompts ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
        3.  [E.5.3 iRAG Prompt](https://arxiv.org/html/2504.21239v1#A5.SS5.SSS3 "In E.5 Evaluation Prompts ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")
11.  [F Example](https://arxiv.org/html/2504.21239v1#A6 "In Memorization and Knowledge Injection in Gated LLMs")
    1.  [F.1 Fictional Character Dataset](https://arxiv.org/html/2504.21239v1#A6.SS1 "In Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")
    2.  [F.2 Wikipedia Dataset](https://arxiv.org/html/2504.21239v1#A6.SS2 "In Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")
    3.  [F.3 Recall Output Examples](https://arxiv.org/html/2504.21239v1#A6.SS3 "In Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")
    4.  [F.4 Questions related to two samples](https://arxiv.org/html/2504.21239v1#A6.SS4 "In Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")
        1.  [F.4.1 Fictional character Compositional Example](https://arxiv.org/html/2504.21239v1#A6.SS4.SSS1 "In F.4 Questions related to two samples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")
        2.  [F.4.2 Wikipedia 2024 Events Compositional Example](https://arxiv.org/html/2504.21239v1#A6.SS4.SSS2 "In F.4 Questions related to two samples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")

Memorization and Knowledge Injection in Gated LLMs
==================================================

Xu Pan    Ely Hahami    Zechen Zhang    Haim Sompolinsky

###### Abstract

Large Language Models (LLMs) currently struggle to sequentially add new memories and integrate new knowledge. These limitations contrast with the human ability to continuously learn from new experiences and acquire knowledge throughout life. Most existing approaches add memories either through large context windows or external memory buffers (e.g., Retrieval-Augmented Generation), and studies on knowledge injection rarely test scenarios resembling everyday life events. In this work, we introduce a continual learning framework, Memory Embedded in Gated LLMs (MEGa), which injects event memories directly into the weights of LLMs. Each memory is stored in a dedicated set of gated low-rank weights. During inference, a gating mechanism activates relevant memory weights by matching query embeddings to stored memory embeddings. This enables the model to both recall entire memories and answer related questions. On two datasets - fictional characters and Wikipedia events - MEGa outperforms baseline approaches in mitigating catastrophic forgetting. Our model draws inspiration from the complementary memory system of the human brain.

Machine Learning, ICML

  

1 Introduction
--------------

The rapid advancement of large language models (LLMs) has dramatically reshaped our understanding of the capabilities and potential of AI systems. These models have proven invaluable across diverse fields, assisting researchers and practitioners. For cognitive neuroscientists, LLMs offer an unprecedented opportunity to study an intelligent system that, although not human, can process natural language and advanced cognitive functions, and compare it to human cognition as well as its underlying brain mechanisms (Hagendorff, [2023](https://arxiv.org/html/2504.21239v1#bib.bib25); Binz & Schulz, [2024](https://arxiv.org/html/2504.21239v1#bib.bib4); Coda-Forno et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib12)). Such comparisons advance human cognitive neuroscience and may lead to more powerful AI systems (Silver & Sutton, [2025](https://arxiv.org/html/2504.21239v1#bib.bib72)). In this paper, we propose to study long-term declarative memory (e.g. episodic memory and semantic memory), one of the hallmarks of human cognition, using LLMs as a model cognitive system. To achieve this, we augment a pretrained LLM with gated memory modules, enabling rapid continual encoding and retrieval of memories while mitigating catastrophic forgetting.

Classical models of long-term memory in neural networks are based on the paradigm of associative memory in recurrent neural networks (RNNs), such as the Hopfield model (Hopfield, [1982](https://arxiv.org/html/2504.21239v1#bib.bib29)), where each memory corresponds to a stable activation pattern of the network. This paradigm was later extended to the memorization of sequences of states (Kleinfeld & Sompolinsky, [1988](https://arxiv.org/html/2504.21239v1#bib.bib40); Kanter & Sompolinsky, [1987](https://arxiv.org/html/2504.21239v1#bib.bib38)). A common feature of these models is the use of Hebbian-like learning rules, which are inherently incremental and align with the continual nature of long-term memory. However, Hebbian learning is severely limited in that it can only store random, uncorrelated patterns, with accurate recall possible only when the number of memories does not exceed a capacity limit that scales linearly with the network size. Beyond this limit, the system suffers from catastrophic forgetting (CF) (French, [1999](https://arxiv.org/html/2504.21239v1#bib.bib17)). When the memorized states are correlated, the capacity is drastically reduced to just a few memories—regardless of network size—due to strong interference between them (Löwe, [1998](https://arxiv.org/html/2504.21239v1#bib.bib48)). Some learning rules partially address this limitation but require batch learning of all memories (Kanter & Sompolinsky, [1987](https://arxiv.org/html/2504.21239v1#bib.bib38); Gardner & Derrida, [1988](https://arxiv.org/html/2504.21239v1#bib.bib19)), which is unsuitable for modeling continual learning (CL) of memories. Moreover, even these batch models of attractor networks degrade significantly when the stored patterns are analog, rather than binary (Schönsberg et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib66)). The challenge of mitigating CF in CL, especially when dealing with realistic, highly correlated analog data, remains a persistent obstacle in both machine learning and cognitive neuroscience. Most current regularization approaches to CL perform poorly when faced with sequences of correlated tasks (Shan et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib69)). Rehearsal-based methods demand substantial memory resources, and existing models of spontaneous memory reactivation in Hebbian RNNs (Shaham et al., [2022](https://arxiv.org/html/2504.21239v1#bib.bib68)) offer limited effectiveness when the memories are correlated.

Another fundamental limitation of classical memory models in “shallow” RNNs is their tendency to encode memories as isolated knowledge items. In contrast, real-world memories are composed of events rich in semantic structure, with elements that are typically already familiar to the organism. As a result, new factual memories must be embedded within or interact closely with an existing, fully developed semantic system. The same principle applies to working memory, where ongoing experiences must integrate with extensive semantic knowledge to support current perceptual or motor tasks. Most neural network models of working memory, however, store items in transient activity patterns that are independent of semantic content (Hochreiter & Schmidhuber, [1997](https://arxiv.org/html/2504.21239v1#bib.bib28)). Although several cognitive theories have proposed memory systems that use pointers or binding mechanisms to associate memory items with their context (Norris, [2017](https://arxiv.org/html/2504.21239v1#bib.bib59); Blouw et al., [2016](https://arxiv.org/html/2504.21239v1#bib.bib5)), no existing model addresses these fundamental challenges at a scale that matches human memory. Finally, a key function of attractor models is pattern completion, achieved through the convergence of all states within a memory’s basin of attraction to a fixed point—a property known as content-addressable memory. Yet, this concept requires substantial revision, as the recall of semantically rich memories typically does not begin from randomly corrupted inputs.

Several studies have explored memory functions in large language models (LLMs), both for AI applications and as analogies to human memory systems (Raccah et al., [2022](https://arxiv.org/html/2504.21239v1#bib.bib64); Wang et al., [2024b](https://arxiv.org/html/2504.21239v1#bib.bib80); Janik, [2023](https://arxiv.org/html/2504.21239v1#bib.bib33); Gutiérrez et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib24); Fountas et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib16); Gershman et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib21); Silver & Sutton, [2025](https://arxiv.org/html/2504.21239v1#bib.bib72)). Within LLMs, three primary analogies to memory have emerged: (1) tokens in the context window, (2) an external database connected to the model, and (3) the model’s internal weights. Each of these representations reflects certain aspects of human memory, yet all fall short of fully modeling human long-term memory.

In-context learning (Brown et al., [2020](https://arxiv.org/html/2504.21239v1#bib.bib6)) appears to avoid the limitations of classical associative memory models: it does not suffer from catastrophic forgetting and can learn new memories, even when they are correlated with existing ones, by smoothly integrating them into the LLM’s existing semantic knowledge. However, the demands of long-term memory may exceed the capacity of context windows (Bai et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib1)). Notably, the context window more closely resembles human working memory. Attempting to unify working and long-term memory into a single representation is a biologically implausible model of memory, as these two functions rely on distinct cognitive resources and brain systems (Fountas et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib16)).

Retrieving memories from an external database into the LLM’s context window is known as Retrieval-Augmented Generation (RAG). RAG has proven to be an effective method for enhancing the memory capabilities of LLMs and is widely used in real-world applications (Lewis et al., [2020](https://arxiv.org/html/2504.21239v1#bib.bib45)). In this approach, user queries are processed by an embedding model to retrieve relevant entries—based on a similarity metric—from the database. These entries are then inserted into the LLM’s context window, enabling the model to generate informed responses by combining its inherent capabilities with the retrieved knowledge.

Studies have also investigated RAG in continual learning settings, where external databases are incrementally updated over time while preserving knowledge consistency (Li et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib46); Gupta et al., [2024b](https://arxiv.org/html/2504.21239v1#bib.bib23); Modarressi et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib56); Zhang et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib90); Fan et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib14)). Variants of RAG have also been developed that store and retrieve key-value pairs (i.e., KV-cache) or other compressed representations of knowledge, rather than raw text (Modarressi et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib56); Das et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib13); Xiao et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib81); Yang et al., [2024b](https://arxiv.org/html/2504.21239v1#bib.bib85); Qian et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib63); Chan et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib8)).

Despite RAG’s usefulness in applications, storing memories in an external text-based database is not an attractive candidate for modeling human long-term memory, since this approach more accurately resembles an external environment with which a subject interacts. Rather, injecting memories by fine-tuning the LLM parameters more closely mirrors the biological process where long-lasting synaptic changes of the neural circuits sustain long-term memory. However, fine-tuning LLMs for long-term memory faces several challenges, such as catastrophic forgetting (Luo et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib50); Chen et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib9); Zhai et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib89); Gupta et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib22); Zhang & Wu, [2024](https://arxiv.org/html/2504.21239v1#bib.bib93); Zhu et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib95); Yang et al., [2024d](https://arxiv.org/html/2504.21239v1#bib.bib87); Song et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib73)), lack of generalization (Berglund et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib2); Ovadia et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib60); Yang et al., [2024c](https://arxiv.org/html/2504.21239v1#bib.bib86)), slow learning and hallucination (Gekhman et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib20); Kang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib37)).

To address these challenges, we introduce MEGa (Memory Embedded in Gated LLMs), a long-term memory framework designed to enable LLMs to sequentially store new memories in a manner that reflects key aspects of human memory.

To ensure biological plausibility, MEGa encodes new memories by fine-tuning the network’s weights. To mitigate catastrophic forgetting, it employs a gating mechanism that, at inference time, routes input queries to a collection of gated memory modules and activates those most relevant to the query.

We show that MEGa is capable not only of retrieving the learned memories but also of performing question-answering (QA) tasks based on them, demonstrating the successful integration of the memories into the knowledge base of the LLM.

Across two datasets—Fictional Character and Wikipedia 2024 Events—and two tasks—memory recall and question answering—MEGa outperforms baseline continual learning (CL) techniques in mitigating both the forgetting of newly acquired memories and the degradation of general language capabilities. These results suggest that MEGa is a promising model for capturing certain aspects of human long-term memory and the structure of underlying brain circuits.

2 Related Works
---------------

### 2.1 Continual Learning and Catastrophic Forgetting

Several studies have shown that LLMs suffer from catastrophic forgetting of previously acquired knowledge and skills during continual learning (CL) on new tasks (Luo et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib50); Chen et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib9); Zhai et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib89); Song et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib73); Zhang & Wu, [2024](https://arxiv.org/html/2504.21239v1#bib.bib93); Zhu et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib95)). To address this issue, various CL methodologies have been adapted for use with LLMs, including regularization-based approaches (Lee et al., [2019](https://arxiv.org/html/2504.21239v1#bib.bib43); Zhang et al., [2020](https://arxiv.org/html/2504.21239v1#bib.bib92); Chen et al., [2020](https://arxiv.org/html/2504.21239v1#bib.bib10); Kotha et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib41); Zhu et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib95)), rehearsal-based strategies (Sun et al., [2019](https://arxiv.org/html/2504.21239v1#bib.bib74); Xu et al., [2024b](https://arxiv.org/html/2504.21239v1#bib.bib83); Huang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib32)), and architecture-based techniques (Hartvigsen et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib27)). Our proposed method MEGa can be considered an architecture-based method, as it adds new components and gating to an existing model.

### 2.2 Knowledge Injection

Injecting new knowledge into pretrained LLMs has recently garnered significant attention (Hsueh et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib30); Shi et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib71); Zhang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib91); Thede et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib76)). A straightforward approach involves fine-tuning the model on the knowledge text (Ovadia et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib60); Gangadhar & Stratos, [2024](https://arxiv.org/html/2504.21239v1#bib.bib18)), or on the answers when the knowledge is provided in the form of QA pairs (Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52)). More recent methods aim to localize weight updates by identifying a knowledge-relevant subspace of the model’s weights (Meng et al., [2022a](https://arxiv.org/html/2504.21239v1#bib.bib53); Mitchell et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib55)), or by distilling knowledge from the context window into the model’s parameters (Qi et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib62); Padmanabhan et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib61); [Wang et al.,](https://arxiv.org/html/2504.21239v1#bib.bib79) ; Kujanpää et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib42)). However, there is evidence that these approaches are not significantly more effective than standard fine-tuning (Gangadhar & Stratos, [2024](https://arxiv.org/html/2504.21239v1#bib.bib18); Thede et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib76)).

However, many of these methods—and their associated experimental setups—fall short of emulating how humans acquire long-term episodic memories. Most widely used knowledge editing datasets (Levy et al., [2017](https://arxiv.org/html/2504.21239v1#bib.bib44); Meng et al., [2022a](https://arxiv.org/html/2504.21239v1#bib.bib53); Zhang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib91); Thede et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib76)) represent knowledge as simple subject-relation-object triples. This format, however, lacks the richness and complexity of human episodic memory, which typically involves detailed semantic representations of personal experiences. As a result, methods developed using subject-relation-object datasets (Meng et al., [2022a](https://arxiv.org/html/2504.21239v1#bib.bib53), [b](https://arxiv.org/html/2504.21239v1#bib.bib54); Mitchell et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib55); Wang et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib78); Qi et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib62); Yu et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib88)) are often not directly applicable for injecting knowledge expressed in paragraph-level event descriptions. Another limitation is that success in these methods is usually defined as the model generating the correct object given a subject-relation pair—whereas human episodic memory is far more flexible: a single memory can be triggered by a variety of cues and used to answer diverse questions.

### 2.3 Gating Networks

Our model, MEGa, uses gating units to route queries to the most relevant stored memories. In general, gating networks function by selectively activating or suppressing connection paths based on the context or input provided to the system. Both empirical studies (Hochreiter & Schmidhuber, [1997](https://arxiv.org/html/2504.21239v1#bib.bib28); Chung et al., [2014](https://arxiv.org/html/2504.21239v1#bib.bib11); Sezener et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib67); Veness et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib77)) and theoretical analyses (Saxe et al., [2022](https://arxiv.org/html/2504.21239v1#bib.bib65); Li & Sompolinsky, [2022](https://arxiv.org/html/2504.21239v1#bib.bib47)) have shown that gated architectures are effective at mitigating catastrophic forgetting and are well-suited for training across multiple tasks. Gating mechanisms are widely used in modern deep neural networks. One prominent example is the Mixture of Experts (MoEs) architecture, a type of gated network that has gained popularity and contributes to some of the state-of-the-art LLMs (Shazeer et al., [2017](https://arxiv.org/html/2504.21239v1#bib.bib70); Fedus et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib15); Jiang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib34)).

Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning technique, which freezes a pretrained weight matrix WPTsubscript𝑊PTW\_{\\text{PT}}italic\_W start\_POSTSUBSCRIPT PT end\_POSTSUBSCRIPT and injects trainable rank decomposition matrices into each layer of the Transformer architecture, such that WFT:=WPT+A⁢Bassignsubscript𝑊FTsubscript𝑊PT𝐴𝐵W\_{\\text{FT}}:=W\_{\\text{PT}}+ABitalic\_W start\_POSTSUBSCRIPT FT end\_POSTSUBSCRIPT := italic\_W start\_POSTSUBSCRIPT PT end\_POSTSUBSCRIPT + italic\_A italic\_B (Hu et al., [2021](https://arxiv.org/html/2504.21239v1#bib.bib31)). Here, WFT∈ℝd×ksubscript𝑊FTsuperscriptℝ𝑑𝑘W\_{\\text{FT}}\\in\\mathbb{R}^{d\\times k}italic\_W start\_POSTSUBSCRIPT FT end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_k end\_POSTSUPERSCRIPT denotes the fine-tuned weight matrix, A∈ℝd×r𝐴superscriptℝ𝑑𝑟A\\in\\mathbb{R}^{d\\times r}italic\_A ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_r end\_POSTSUPERSCRIPT and B∈ℝr×k𝐵superscriptℝ𝑟𝑘B\\in\\mathbb{R}^{r\\times k}italic\_B ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_r × italic\_k end\_POSTSUPERSCRIPT are trainable update matrices, and d𝑑ditalic\_d and k𝑘kitalic\_k are input and output dimensions.

Several studies have explored using gated LoRA modules to enhance fine-tuning performance across multiple tasks. Some approaches determine per-token gating weights based on routing networks (Jung & Kim, [2024](https://arxiv.org/html/2504.21239v1#bib.bib35); Buehler & Buehler, [2024](https://arxiv.org/html/2504.21239v1#bib.bib7); Luo et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib49); Xu et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib82); Zhao et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib94)), while others rely on local activation levels (Wang et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib78); Zhu et al., [2024b](https://arxiv.org/html/2504.21239v1#bib.bib96)). In contrast, our work determines gating weights based on the semantic similarity between the query and the stored knowledge, which serves as a global signal for all the layers and tokens. The MELO framework (Yu et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib88)) is the most comparable to MEGa in this respect, although our datasets and evaluations are more complex and better capture the characteristics of human long-term episodic memory.

![Refer to caption](extracted/6399726/Illustration.png)

  

Figure 1: Illustration of our fine-tuning and inference pipeline. We store the embedding of each sample as its context key and initialize a LoRA adapter for each sample (i.e., raw text paragraphs). LoRA adapters are fine-tuned individually using a constructed user prompt. During inference, the base model processes the query to generate its embedding, which is then used to compute the LoRA gating weights. Questions can be augmented with additional instructions depending on the task. For the recall task, the model is prompted to reconstruct the entire relevant story. For the QA task, the model directly answers the question, with the instruction “Answer should be no more than one sentence” appended to ensure concise responses. In the internal RAG (iRAG) setup, the model first reconstructs the relevant story and then answers the question based on the reconstructed story. All inputs are formatted according to the Llama-3 instruct format.

3 Methods
---------

### 3.1 Problem Formulation

Our goal is to build a system capable of memorizing multiple memories that arrive sequentially, and later retrieving those memories and extracting knowledge from them—while minimizing destructive interference both among the memories themselves and across the model’s general language abilities. This mirrors how the human brain continuously acquires declarative memory through experience. Individual memories are rich in semantic context, such as events involving people, actions, time, and locations.

The datasets used in this work consist of a set {D1,D2,…⁢Dn}subscript𝐷1subscript𝐷2…subscript𝐷𝑛\\{D\_{1},D\_{2},...D\_{n}\\}{ italic\_D start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT , italic\_D start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT , … italic\_D start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT } where each sample, Disubscript𝐷𝑖D\_{i}italic\_D start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT, is a short paragraph describing an event that the model has not seen before. These memories are learned by a pretrained LLM through fine-tuning. Importantly, rather than being presented in a batch, the samples are provided to the model sequentially and exclusively—one at a time—reflecting both the nature of human experience and real-world scenarios where task data arrives incrementally. The goal is for the model, after fine-tuning, to (1) reconstruct each individual sample when given an relevant cue, and (2) answer questions related to one or more of the fine-tuned texts. We refer to the first as the recall task and the second as the QA task, corresponding to the model’s memory and knowledge, respectively.

  

Fictional character

Wiki events

QA Acc % ↑↑\\uparrow↑

Log prob ↓↓\\downarrow↓

Recall cos ↑↑\\uparrow↑

MMLU acc % ↑↑\\uparrow↑

QA Acc % ↑↑\\uparrow↑

Log prob ↓↓\\downarrow↓

Recall cos ↑↑\\uparrow↑

MMLU acc %↑↑\\uparrow↑

Base

0.13±0.27

\-4.85±0.39

0.587±0.024

62.56

8.17±2.38

\-3.25±0.14

0.665±0.018

62.56

RAG

82.57±6.37

\-2.57±0.46

0.881±0.002

62.56

88.83±2.71

\-1.76±0.15

0.889±0.006

62.56

Full

12.60±11.03

\-2.92±0.30

0.554±0.066

55.65±1.54

17.90±9.57

\-2.00±0.28

0.526±0.149

56.28±1.03

Full L2

25.47±10.32

\-3.14±0.45

0.609±0.084

55.25±1.79

31.33±11.69

\-2.07±0.33

0.623±0.163

55.98±1.48

Full ewc

14.10±8.08

\-2.82±0.29

0.542±0.064

55.24±1.75

21.43±13.47

\-2.02±0.32

0.544±0.149

56.21±0.98

Full batch

54.77±3.27

\-2.77±0.28

0.853±0.013

60.62±1.05

76.87±5.72

\-1.87±0.26

0.926±0.011

60.90±1.34

Lora

0.80±1.20

\-4.13±0.44

0.485±0.038

47.94±2.11

0.53±0.72

\-3.02±0.49

0.243±0.055

46.88±1.42

Lora L2

12.27±6.98

\-3.30±0.42

0.528±0.058

47.49±2.53

18.20±9.81

\-2.35±0.41

0.454±0.144

49.12±2.30

MEGa (ours)

72.53±6.79

\-2.12±0.25

0.901±0.011

61.75±0.53

78.03±3.50

\-1.86±0.18

0.921±0.013

61.99±0.57

iRAG (ours)

80.67±6.09

\-2.84±0.33

0.901±0.011

61.75±0.53

84.70±4.37

\-2.24±0.21

0.921±0.013

61.99±0.57

Table 1: Evaluation after continual learning on 50 samples. QA accuracy (evaluated by a GPT judge) and log probability assess the model’s question-answering ability, while recall cosine similarity measures memorization performance. MMLU accuracy reflects the degree of catastrophic forgetting in the model’s general knowledge. Results are reported as the mean and standard deviation across 20 dataset partitions. For RAG in the reconstruction task, we report the metric: hit rate⋅1+(1−hit rate)⋅c,⋅hit rate1⋅1hit rate𝑐\\text{hit rate}\\cdot 1+(1-\\text{hit rate})\\cdot c,hit rate ⋅ 1 + ( 1 - hit rate ) ⋅ italic\_c , where c𝑐citalic\_c is the average cosine similarity between randomly selected training samples, and the hit rate is the percentage of times RAG includes the correct sample in the LLM’s context window. The MMLU accuracy of RAG is reported same as the base model.

### 3.2 Model and Datasets

All experiments are conducted using the Llama-3.1-8B-Instruct model, referred to as the “base” model. We fine-tune it on two datasets:

Fictional Character Dataset: We generate synthetic data by prompting GPT-4.5 (see Appendix [E](https://arxiv.org/html/2504.21239v1#A5 "Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")) to produce 50 paragraphs (i.e., samples) based on a manually created template. Each paragraph describes a specific event in the life of a fictional character with a randomly generated name. To align with our goal of modeling episodic memory, the paragraphs capture concrete events (e.g., a basketball game, a trip to the Swiss Alps). This procedure is repeated 20 times with different random names, resulting in 20 dataset partitions. Continual learning experiments are performed separately on each partition, and unless otherwise noted, reported evaluation metrics include the mean and standard deviation across the 20 partitions. The average length of each story is 41.93 words. Following prior work (Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52); Mosbach et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib57); Ovadia et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib60)), we also generate nine paraphrases for each sample during fine-tuning using GPT-4.5.

Lastly, to construct the evaluation set, we prompt GPT-4.5 to generate three QA pairs for each sample, based on its content (see Appendix [E](https://arxiv.org/html/2504.21239v1#A5 "Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs") for prompt details and Appendix [F.1](https://arxiv.org/html/2504.21239v1#A6.SS1 "F.1 Fictional Character Dataset ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs") for an example).

Wikipedia 2024 Events: Following previous knowledge-injection studies (Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52); Ovadia et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib60); Zhang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib91)), we use Wikipedia articles on recent events as our second dataset. Since the knowledge cutoff date for Llama-3.1-8B-Instruct is December 2023, we crawled all Wikipedia articles categorized under “2024\_by\_month” and its subcategories. We further filtered out articles that were first created before 2024. For fine-tuning, we use the first section of each retained article, which typically provides a summary of the event. From these, we randomly sampled 1,000 articles with character counts between 200 and 300 (average word count: 41.55) to form our fine-tuning dataset. As with the Fictional Character dataset, we partition these samples into 20 subsets for continual learning experiments and generate corresponding paraphrases and QA pairs (see Appendix [F.2](https://arxiv.org/html/2504.21239v1#A6.SS2 "F.2 Wikipedia Dataset ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs") for an example).

Compositional questions: In addition to QA pairs based on individual samples, we prompt GPT-o3-mini (see full prompt in Appendix [E.2](https://arxiv.org/html/2504.21239v1#A5.SS2 "E.2 Compositional Question Dataset Construction ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")) to generate an evaluation set of 500 compositional questions—each requiring knowledge from exactly two distinct samples to answer correctly (see Appendix [F.4](https://arxiv.org/html/2504.21239v1#A6.SS4 "F.4 Questions related to two samples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs") for an example).

### 3.3 Memory Embedded in Gated LLMs (MEGa)

The fine-tuning and inference algorithm of MEGa is shown schematically in Figure [1](https://arxiv.org/html/2504.21239v1#S2.F1 "Figure 1 ‣ 2.3 Gating Networks ‣ 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs").

#### 3.3.1 Fine-tuning

During fine-tuning on a new sample, Disubscript𝐷𝑖D\_{i}italic\_D start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT, we first get and store the embedding of this sample as “context key”, Ki\=f⁢(Di)subscript𝐾𝑖𝑓subscript𝐷𝑖K\_{i}=f(D\_{i})italic\_K start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = italic\_f ( italic\_D start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ), where f𝑓fitalic\_f is an embedding function that maps a sequence of tokens into a fixed-sized vector containing semantic-rich information about the text. We employ embeddings generated internally by the base model. Specifically, we define f𝑓fitalic\_f as the average of the internal activations computed near the end of the base model—precisely, the input to the final MLP layer (Muennighoff, [2022](https://arxiv.org/html/2504.21239v1#bib.bib58)). We explore various embedding layers and embedding models in Appendix [C](https://arxiv.org/html/2504.21239v1#A3 "Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs").

Next, we fine-tune a set of weights denoted as LoRA adapter. Each LoRA adapter consists of a set of low-rank matrices, {Al,Bl}superscript𝐴𝑙superscript𝐵𝑙\\{A^{l},B^{l}\\}{ italic\_A start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT , italic\_B start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT }, applied on specified layers {l}𝑙\\{l\\}{ italic\_l } and modules. For each sample, our gated-LoRA method initializes and trains exactly one LoRA adapter. Specifically, when training on the i𝑖iitalic\_ith sample Disubscript𝐷𝑖D\_{i}italic\_D start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT, {Ail,Bil}superscriptsubscript𝐴𝑖𝑙superscriptsubscript𝐵𝑖𝑙\\{A\_{i}^{l},B\_{i}^{l}\\}{ italic\_A start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT , italic\_B start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT } are (see the Appendix [D](https://arxiv.org/html/2504.21239v1#A4 "Appendix D Fine-Tuning Settings ‣ Memorization and Knowledge Injection in Gated LLMs") on initialization) added to the pre-trained weights WP⁢Tlsubscriptsuperscript𝑊𝑙𝑃𝑇W^{l}\_{PT}italic\_W start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_P italic\_T end\_POSTSUBSCRIPT to form the new set of weights ΘisubscriptΘ𝑖\\Theta\_{i}roman\_Θ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT:

fine-tuning weights:⁢Θi\={WP⁢Tl+Ail⁢Bil}.fine-tuning weights:subscriptΘ𝑖subscriptsubscriptsuperscript𝑊𝑙𝑃𝑇subscriptsuperscript𝐴𝑙𝑖subscriptsuperscript𝐵𝑙𝑖.\\text{{fine-tuning weights:}}\\>\\>\\Theta\_{i}=\\{W^{l}\_{PT}+A^{l}\_{i}B^{l}\_{i}\\}\_% {.}fine-tuning weights: roman\_Θ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = { italic\_W start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_P italic\_T end\_POSTSUBSCRIPT + italic\_A start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT italic\_B start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT } start\_POSTSUBSCRIPT . end\_POSTSUBSCRIPT

(1)

The LoRA adapter weights are fine-tuned by minimizing the following loss,

Li⁢({Ail,Bil})\=−log⁡pΘi⁢(Di∣x),subscript𝐿𝑖subscriptsuperscript𝐴𝑙𝑖subscriptsuperscript𝐵𝑙𝑖subscript𝑝subscriptΘ𝑖conditionalsubscript𝐷𝑖𝑥L\_{i}(\\{A^{l}\_{i},B^{l}\_{i}\\})=-\\log p\_{\\Theta\_{i}}(D\_{i}\\mid x),italic\_L start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( { italic\_A start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT , italic\_B start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT } ) = - roman\_log italic\_p start\_POSTSUBSCRIPT roman\_Θ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( italic\_D start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∣ italic\_x ) ,

(2)

Here, x𝑥xitalic\_x refers to an appropriate fine-tuning prompt. Common choices include the “begin of text” token (i.e., continued pretraining) and formatted user queries (i.e., supervised fine-tuning). We found that using the “begin of text” token resulted in poor performance on the Fictional Character dataset (see Appendix [B](https://arxiv.org/html/2504.21239v1#A2 "Appendix B Supervised Fine-Tuning ‣ Memorization and Knowledge Injection in Gated LLMs")). For our main experiments, we use the user query prompt “Tell me a story that you memorized.” (see Figure [1](https://arxiv.org/html/2504.21239v1#S2.F1 "Figure 1 ‣ 2.3 Gating Networks ‣ 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs") and Appendix [B](https://arxiv.org/html/2504.21239v1#A2 "Appendix B Supervised Fine-Tuning ‣ Memorization and Knowledge Injection in Gated LLMs")). Other prompt formats are also possible (see Appendix [B](https://arxiv.org/html/2504.21239v1#A2 "Appendix B Supervised Fine-Tuning ‣ Memorization and Knowledge Injection in Gated LLMs") for details).

For our main experiments, we target all MLP layers and use rank r\=128𝑟128r=128italic\_r = 128.The impact of selecting different layers and modules for fine-tuning is detailed in Appendix [C](https://arxiv.org/html/2504.21239v1#A3 "Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs").

#### 3.3.2 Inference

During inference, we add the weighted sum of A⁢s𝐴𝑠Asitalic\_A italic\_s and B⁢s𝐵𝑠Bsitalic\_B italic\_s to the pretrained weights

inference weights:⁢Θi⁢n⁢f⁢e⁢rl\={WP⁢Tl+∑igi⁢Ail⁢Bil},inference weights:subscriptsuperscriptΘ𝑙𝑖𝑛𝑓𝑒𝑟subscriptsuperscript𝑊𝑙𝑃𝑇subscript𝑖subscript𝑔𝑖subscriptsuperscript𝐴𝑙𝑖subscriptsuperscript𝐵𝑙𝑖\\text{{inference weights:}}\\>\\>\\Theta^{l}\_{infer}=\\{W^{l}\_{PT}+\\sum\_{i}g\_{i}A^% {l}\_{i}B^{l}\_{i}\\},inference weights: roman\_Θ start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i italic\_n italic\_f italic\_e italic\_r end\_POSTSUBSCRIPT = { italic\_W start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_P italic\_T end\_POSTSUBSCRIPT + ∑ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT italic\_g start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT italic\_A start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT italic\_B start\_POSTSUPERSCRIPT italic\_l end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT } ,

(3)

where gisubscript𝑔𝑖g\_{i}italic\_g start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT are gating weights for each LoRA adapter. The gating weights are computed per user query and serve as a global signal shared across all the layers. We compute the gating weights by comparing the user’s query embedding f⁢(q)𝑓𝑞f(q)italic\_f ( italic\_q ) with all context keys K\=\[K1⁢K2⁢…⁢Kn\]Kdelimited-\[\]subscript𝐾1subscript𝐾2…subscript𝐾𝑛\\textbf{K}=\[K\_{1}\\>K\_{2}\\>...\\>K\_{n}\]K = \[ italic\_K start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT italic\_K start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT … italic\_K start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT \] where n𝑛nitalic\_n is the total number of memories.

𝒈\=softmax⁢(β⁢f⁢(q)⊤⁢K),𝒈softmax𝛽𝑓superscript𝑞topK\\bm{g}=\\text{softmax}(\\beta f(q)^{\\top}\\textbf{K}),bold\_italic\_g = softmax ( italic\_β italic\_f ( italic\_q ) start\_POSTSUPERSCRIPT ⊤ end\_POSTSUPERSCRIPT K ) ,

(4)

Here β𝛽\\betaitalic\_β is a parameter controlling how spread the gating weights are. We set β\=1𝛽1\\beta=1italic\_β = 1 for the main experiments, and β\=0.1𝛽0.1\\beta=0.1italic\_β = 0.1 for the compositional question experiments.

As mentioned in the previous subsection, we choose f𝑓fitalic\_f as the average of the token-level internal activation vectors. Since the input must be processed by the base model to generate a response, extracting the embedding from these internal activations incurs no additional computational cost beyond standard inference. Moreover, this design choice ensures that both fine-tuning and retrieval remain fully encapsulated within a single, unified model.

Given a query q𝑞qitalic\_q the model generates a sequence of tokens a0:tsubscript𝑎:0𝑡a\_{0:t}italic\_a start\_POSTSUBSCRIPT 0 : italic\_t end\_POSTSUBSCRIPT by iteratively sampling tokens from

pΘi⁢n⁢f⁢e⁢r⁢(at∣a0:t−1,q).subscript𝑝subscriptΘ𝑖𝑛𝑓𝑒𝑟conditionalsubscript𝑎𝑡subscript𝑎:0𝑡1𝑞p\_{\\Theta\_{infer}}(a\_{t}\\mid a\_{0:t-1},q).italic\_p start\_POSTSUBSCRIPT roman\_Θ start\_POSTSUBSCRIPT italic\_i italic\_n italic\_f italic\_e italic\_r end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( italic\_a start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ∣ italic\_a start\_POSTSUBSCRIPT 0 : italic\_t - 1 end\_POSTSUBSCRIPT , italic\_q ) .

(5)

In the main experiments, we use the greedy sampling strategy. Depending on the user query q𝑞qitalic\_q, the model can do different tasks, such as recalling a memory or answering a question related to the memories.

Memory Recall: When testing the model’s memorization, we append memory-related questions with an extra prompt “Reconstruct the entire story that is related to the above question.”. The model is expected to generate the entire relevant story, rather than answering the question (Figure [1](https://arxiv.org/html/2504.21239v1#S2.F1 "Figure 1 ‣ 2.3 Gating Networks ‣ 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")).

QA: In QA tasks, we found that sometimes the generated answer is too long. To ensure the generation quality and not to confound the question-answering ability with memorization ability, in QA tasks an extra prompt “Answer should be no more than one sentence.” is appended to each question (Figure [1](https://arxiv.org/html/2504.21239v1#S2.F1 "Figure 1 ‣ 2.3 Gating Networks ‣ 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")).

Internal RAG (iRAG): Since the model can recall relevant memories in response to a question, it should also be able to leverage the recalled content to assist in answering the question. We refer to this approach as internal RAG (iRAG). Specifically, given a question, we first append the memory recall prompt to retrieve the full memory associated with the question. The question is then presented to the model again, along with the additional prompt: “Based on the reconstructed story, answer the following question: {QUESTION} Answer should be no more than one sentence.” iRAG resembles traditional RAG in that it places the relevant knowledge directly into the context window and utilizes in-context learning. However, unlike RAG, the knowledge in iRAG is generated internally by the model rather than retrieved from an external database (see Figure [1](https://arxiv.org/html/2504.21239v1#S2.F1 "Figure 1 ‣ 2.3 Gating Networks ‣ 2 Related Works ‣ Memorization and Knowledge Injection in Gated LLMs")). iRAG can be viewed as a form of chain-of-thought reasoning. Additionally, it resembles the function of memory in traditional associative memory models. For instance, in attractor models long term memories are first being activated by convergence to the fixed point and then uploaded to a working memory module to be further processed according to the demands of the ongoing tasks.

### 3.4 Evaluation

#### 3.4.1 Memory Recall

We evaluate the quality of the recalled memory by computing the cosine similarity between the embeddings (generated by OpenAI text-embedding-3-large) of the model output and the original sample. We refer to it as recall cos.

#### 3.4.2 QA

In the QA task, we do not require the generated answer to exactly match the correct answer, as the same information can often be expressed in multiple ways (see Appendix [F](https://arxiv.org/html/2504.21239v1#A6 "Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")). Instead, we use GPT-o3-mini as a judge to determine whether a given response is correct and satisfies the instruction “Answer should be no more than one sentence.” (see Appendix [E.3](https://arxiv.org/html/2504.21239v1#A5.SS3 "E.3 GPT judge ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs")). The judge is prompted to produce a binary result. We also report the widely used log probability (log prob) as an additional metric. However, we argue that the GPT judge provides a more reliable evaluation than log prob. Log probability can be misleading: a higher log prob may suggest that the model is more confident, but it does not guarantee that the correct answer is ranked first or even within the top predictions. Moreover, a high log prob can sometimes reflect model collapse rather than genuine correctness in generating the right answers. In contrast, evaluating greedy generation with the GPT judge ensures that both knowledge acquisition (efficiency) and instruction-following ability (generality) are properly assessed.

#### 3.4.3 General Knowledge Baseline

To assess whether fine-tuning degrades the model’s general knowledge, we evaluate it on the widely used general knowledge QA dataset, Massive Multitask Language Understanding (MMLU). We report macro accuracy, defined as the average accuracy across four categories: abstract algebra, anatomy, astronomy, and business ethics. Answers are generated in a single step, without chain-of-thought reasoning. The prompt format is detailed in Appendix [E](https://arxiv.org/html/2504.21239v1#A5 "Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs").

### 3.5 Continual Learning Baselines

To compare MEGa with non-gated approaches, we include several baselines: continual full-parameter fine-tuning (Full, Full+L2, Full+EWC), continual LoRA fine-tuning (LoRA, LoRA+L2), full-parameter batch learning, and Retrieval-Augmented Generation (RAG).

Continual LoRA entails fine-tuning and merging LoRA adapters into the main weights after each sample (Biderman et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib3)). Among these baselines, L2 and EWC (Elastic Weight Consolidation) (Kirkpatrick et al., [2017](https://arxiv.org/html/2504.21239v1#bib.bib39); Shan et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib69)) are common regularization methods for CL. When applying L2 regularization loss, in full-parameter fine-tuning, the L2 loss of the model weights is calculated relative to the model weights before training on the current sample; in LoRA fine-tuning, L2 loss is calculated using the simple LoRA weights. EWC utilizes the second-order error gradient around previous weights. Since, in the continual LoRA setting, the LoRA weights after training on individual samples are merged into the main weights, it is not applicable to use EWC regularization. Thus we only tested EWC in the full-parameter tuning case. We ran a hyperparameter sweep (see Appendix [C](https://arxiv.org/html/2504.21239v1#A3 "Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")), and chose λL⁢2\=0.1subscript𝜆𝐿20.1\\lambda\_{L2}=0.1italic\_λ start\_POSTSUBSCRIPT italic\_L 2 end\_POSTSUBSCRIPT = 0.1 and λEWC\=1.0subscript𝜆EWC1.0\\lambda\_{\\text{EWC}}=1.0italic\_λ start\_POSTSUBSCRIPT EWC end\_POSTSUBSCRIPT = 1.0 in the main experiments.

We also include full-parameter batch learning (mini batch size 2), where the model is trained simultaneously on all available data. Batch learning performance is often considered the ceiling for continual learning methods. As discussed later, the performance gap between batch learning and MEGa provides insights into the potential benefits of leveraging MEGa’s self-recall ability for rehearsal-based learning.

Additionally, we include RAG, which is typically regarded as the ceiling for fine-tuning-based knowledge injection methods. For a fair comparison, we use the same embedding model as MEGa (Llama embeddings). In the main experiments, the top-1 matched sample is put in the context window to generate the answer.

4 Results
---------

### 4.1 Memory Recall

![Refer to caption](extracted/6399726/main_recall.png)

Figure 2: Recall forgetting curve comparing MEGa and other approaches. The model is prompted to recall the fine-tuned sample associated with a given question. Cosine similarity is computed between the recalled text and the original fine-tuning sample. The “Random pairs” baseline represents the average cosine similarity between embeddings of two randomly selected samples (0.465 for the Fictional Character dataset and 0.226 for the Wiki Events dataset).

Table [1](https://arxiv.org/html/2504.21239v1#S3.T1 "Table 1 ‣ 3.1 Problem Formulation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs") presents the evaluation results for MEGa and the baselines. The metric recall cos measures the model’s ability to memorize the fine-tuning samples (see examples in Appendix [F.3](https://arxiv.org/html/2504.21239v1#A6.SS3 "F.3 Recall Output Examples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")). MEGa excels at recalling relevant memories when cued by questions, achieving a mean cosine similarity of 0.901 on the Fictional Character dataset and 0.921 on the Wikipedia dataset. MEGa selects the correct gate (i.e., the context key with the highest similarity to the question’s source) 85.0% of the time for the Fictional Character dataset and 87.8% of the time for the Wiki Events dataset. When the correct gate is chosen, MEGa recalls memories with high fidelity.

While full-parameter batch learning also performs well on this task (0.853 on the Fictional Character dataset and 0.926 on the Wiki Events dataset), all other continual learning (CL) baselines (Full, Full+L2, Full+EWC, LoRA, LoRA+L2) perform substantially worse. These models frequently default to retrieving the most recently seen memory, regardless of the query (see examples in Appendix [F.3](https://arxiv.org/html/2504.21239v1#A6.SS3 "F.3 Recall Output Examples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs")). The base model often fails by refusing to answer, responding with statements like “I don’t have any information about…,” which, although incorrect, sometimes partially overlap with the fine-tuning samples.

The success of MEGa recalling memories also indicates that our fine-tuning procedure retains the model’s instruction-following ability, so that the model appropriately processes the retrieval prompt.

The source of the gap between the performance of MEGa and other CL fine-tuning methods is the characteristic of catastrophic forgetting. We show the forgetting curves computed by the accuracy of retrieving the first sample in the sequence as a function of the length of the sequence in Figure [2](https://arxiv.org/html/2504.21239v1#S4.F2 "Figure 2 ‣ 4.1 Memory Recall ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs"). All curves exhibit roughly similar good performance initially. However, the CL baselines show severe catastrophic forgetting, although L2 regularization helps mitigate it to a limited extent (see Appendix [4](https://arxiv.org/html/2504.21239v1#A3.T4 "Table 4 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs") for a comprehensive analysis on regularization). In contrast, MEGa shows only mild forgetting. The curve of full-parameter batch learning in Figure [2](https://arxiv.org/html/2504.21239v1#S4.F2 "Figure 2 ‣ 4.1 Memory Recall ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs") is almost flat. The very small decrease in the ability to retrieve the early stories in response to questions is analogous to shrinking the basins of attraction of memories in attractor networks even below memorization capacity.

### 4.2 Question-Answering

In addition to memorizing the samples, MEGa performs well on direct QA tasks, indicating that the model not only retains the content of the memory but also integrates it into its existing knowledge. MEGa achieves a QA accuracy of 72.53% on the Fictional Character dataset and 78.03% on the Wiki dataset, significantly outperforming other continual learning fine-tuning approaches. Its performance is also comparable to that of RAG, which achieves 82.57% and 88.83% on the respective datasets—demonstrating that MEGa’s fine-tuning procedure effectively embeds new memories into the model’s weights.

Notably, the base model—prior to any fine-tuning—shows very low QA accuracy (0.13% on the Fictional Character dataset and 8% on the Wiki dataset), confirming that the training data introduces novel knowledge (see Table [1](https://arxiv.org/html/2504.21239v1#S3.T1 "Table 1 ‣ 3.1 Problem Formulation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs")).

Figure [3](https://arxiv.org/html/2504.21239v1#S4.F3 "Figure 3 ‣ 4.4 General Knowledge Retention ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs") shows the forgetting curves for the different fine-tuning methods. Similar to the memory recall task, all methods initially perform well; however, other continual learning (CL) approaches suffer severe degradation in question-answering ability over time. Full-parameter fine-tuning outperforms LoRA without L2 regularization, likely because weight updates are smaller in norms but distributed across more parameters compared to LoRA. Adding L2 regularization improves both full fine-tuning and LoRA fine-tuning, but both still fall significantly short of MEGa’s performance across both datasets. In full-parameter batch learning, QA accuracy exhibits only mild decay as the number of fine-tuning samples increases. MEGa outperforms full-parameter batch learning on the Fictional Character dataset and performs comparably on the Wiki Events dataset, indicating that MEGa’s gating mechanism effectively mitigates forgetting.

Importantly, because MEGa suppresses interference between different memory traces within the fine-tuned model weights, the residual decline in performance with an increasing number of stored memories is primarily due to failures in the softmax-based gating selection mechanism (see Appendix Figure [8](https://arxiv.org/html/2504.21239v1#A3.F8 "Figure 8 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")). This observation suggests that enhancing the quality of context and query embeddings could further reduce performance degradation. Indeed, we find that employing a superior embedding method substantially reduces forgetting (Appendix Figure [9](https://arxiv.org/html/2504.21239v1#A3.F9 "Figure 9 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")).

### 4.3 Internal RAG

Inspired by RAG’s high QA accuracy and MEGa’s near-perfect ability to recall relevant samples, we explore iRAG as an alternative way to use MEGa for question-answering. In iRAG, the model first recalls the relevant memory and then answers the question based on the recalled content. As shown in Table [1](https://arxiv.org/html/2504.21239v1#S3.T1 "Table 1 ‣ 3.1 Problem Formulation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs") and Figure [3](https://arxiv.org/html/2504.21239v1#S4.F3 "Figure 3 ‣ 4.4 General Knowledge Retention ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs"), iRAG further boosts MEGa’s QA performance, achieving 80.67% accuracy on the Fictional Character dataset and 84.70% on the Wiki Events dataset—effectively closing the performance gap with RAG.

### 4.4 General Knowledge Retention

We also evaluate the potential degradation of the model’s general language capabilities resulting from fine-tuning. After training on both datasets, MEGa maintains MMLU accuracy (61.75% and 61.99%), closely matching the performance of the base model, whereas all other continual learning (CL) fine-tuning methods exhibit a noticeable decline (see Table [1](https://arxiv.org/html/2504.21239v1#S3.T1 "Table 1 ‣ 3.1 Problem Formulation ‣ 3 Methods ‣ Memorization and Knowledge Injection in Gated LLMs") and Figure [4](https://arxiv.org/html/2504.21239v1#S4.F4 "Figure 4 ‣ 4.4 General Knowledge Retention ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")). These results suggest that MEGa effectively integrates new information while preserving prior general knowledge and instruction-following abilities—key features for robust continual learning systems.

![Refer to caption](extracted/6399726/main_acc.png)

Figure 3: QA accuracy forgetting curve comparing MEGa and other approaches. The curve shows the QA accuracy for questions related to the first sample in the sequence, measured as the model is sequentially trained on additional samples. As more samples are introduced, accuracy on the first sample declines. For the Fictional Character dataset, the base model’s QA accuracy is 0.13%, overlapping with the x-axis.

![Refer to caption](extracted/6399726/MMLUcurve.png)

Figure 4: MMLU accuracy during continual fine-tuning.

### 4.5 Composite Knowledge

MEGa can mix different LoRA adapters by applying softmax-weighted gating, enabling the combination of knowledge from multiple memories. To test this capability, we evaluate whether such a mixture can correctly answer questions that require information from two separate samples.

We find that full-parameter batch learning performs best on compositional questions, with MEGa performing comparably. Among continual learning methods, MEGa significantly outperforms all other CL baselines (see Table [2](https://arxiv.org/html/2504.21239v1#S4.T2 "Table 2 ‣ 4.5 Composite Knowledge ‣ 4 Results ‣ Memorization and Knowledge Injection in Gated LLMs")). Example compositional questions and model answers are provided in Appendix [F.4](https://arxiv.org/html/2504.21239v1#A6.SS4 "F.4 Questions related to two samples ‣ Appendix F Example ‣ Memorization and Knowledge Injection in Gated LLMs").

While model merging has been widely used to combine task-specific models for computational efficiency (Yang et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib84); Wang et al., [2024a](https://arxiv.org/html/2504.21239v1#bib.bib78)), our results provide the first demonstration that knowledge embedded in separate LoRA modules can be successfully integrated to answer compositional questions through simple weight merging.

Fictional

Wiki

Methods

Character

dataset

MEGa

49.6%

70.4%

Full

9.6%

20.0%

Full L2

26.0%

35.6%

Full EWC

8.4%

16.4%

Full Batch

54.4%

75.2%

LoRA

0.4%

0.4%

LoRA + L2

8.4%

18.4%

Table 2: Compositional QA Accuracy.

5 Discussion
------------

In this study, we present a framework, MEGa, for sequentially injecting new pieces of knowledge into a pretrained LLM. Our goal is twofold: address current challenges in continual knowledge injections in LLMs and AI agents, and develop LLM-based viable models of human memory functions. Some of the challenges met in this work arise from the features of current LLMs such as sensitivity to specific prompts and the varying quality of embeddings. Others are more general and have implications for any intelligent systems, such as mitigating the risk of catastrophic forgetting (CF). Continual learning remains a challenge for ML systems, in contrast to the impressive abilities of humans as lifelong learners, accumulating new memories and knowledge through sequential experiences with their environment. MEGa injects individual memories into fine-tuned LoRA modules and mitigates CF through an integrated gating mechanism. MEGa achieves performance comparable to, or surpassing, RAG and full-parameter batch learning, which are considered the ceiling performance for knowledge editing and continual learning, respectively. Notably, MEGa stores memories within network weights rather than relying on an external database like RAG, thus providing a more biologically plausible model of human memory.

We have constructed and studied two datasets. The Wiki events dataset is widely adopted in LLM knowledge injection studies (Ovadia et al., [2023](https://arxiv.org/html/2504.21239v1#bib.bib60); Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52); Zhang et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib91); Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52); Thede et al., [2025](https://arxiv.org/html/2504.21239v1#bib.bib76)). These types of memories resemble factual knowledge, a form of human semantic memory. In contrast, the autobiographic nature of the stories in the fictional character dataset makes them candidates for the study of human episodic memories and also crucial for AI personalization, alignment, and agency. Despite their importance, datasets resembling everyday life memories about a person were rarely used in LLM knowledge injection studies.

Not surprisingly, the two datasets have different statistical structures. The fictional character stories are more correlated than the wiki events, hence suffer more from interference in the gating selection operation (Appendix Figure [7](https://arxiv.org/html/2504.21239v1#A3.F7 "Figure 7 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")), even with strong embedding models (Appendix Figure [8](https://arxiv.org/html/2504.21239v1#A3.F8 "Figure 8 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs"), [9](https://arxiv.org/html/2504.21239v1#A3.F9 "Figure 9 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")). Another interesting finding is that fine-tuning later MLP layers is more effective to inject fictional character knowledge; while fine-tuning early MLP layers is more effective for the Wiki event knowledge, suggesting that knowledge from the two datasets, due to their distinct nature, might be located in different layers. This hypothesis needs to be validated by further analysis including ablation experiments.

Our current feasibility study uses relatively small scale memory data. As indicated by our results (Appendix Figure [8](https://arxiv.org/html/2504.21239v1#A3.F8 "Figure 8 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs") [9](https://arxiv.org/html/2504.21239v1#A3.F9 "Figure 9 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs")), scaling up the model is possible if one uses an LLM with better embeddings than the present one. In addition, both in humans and AI agents, episodic memories enter the neural systems as sensory experiences. Our preesent ’autobiographic’ stories should be interpreted as internal summaries of the experiences. Extending our model to incorporate multimodal episodic memories in MEGa is an important future research goal.

One limitation of MEGa is that its parameter count grows linearly with the number of training samples, as each new memory requires additional LoRA adapter. This scalability issue can lead to increased computational and storage demands. This might be partially resolved by using shared LORA weights and post learning pruning methods.

A promising future direction is to gradually distill LoRA weights into the base model weights. This entails a rehearsal process in which gating units are activated repeatedly, generating a spontaneous reconstruction of one or few stored memories at a time, and then triggering a slow fine-tuning of the base model. Our batch baseline finetuning, which uses fine-tuning steps of mini-batches of 2 memories, can be thought of as implementing such a rehearsal process. To complete this transfer, it is important to complement this rehearsal process with a gradual phasing out of some LoRA adapters depending on age, frequency of use, or other saliency criteria. Incorporating rehearsal-based memory transfer will make the model similar to the complementary memory systems hypothesis for human long-term memory. Here, the gating systems and associated LoRA weights correspond to the fast learner (“hippocampus”) while the rehearsal-triggered fine-tuning of the base weights correspond to the slow learner (“cortex”) (McClelland et al., [2020](https://arxiv.org/html/2504.21239v1#bib.bib51)). The gating operation in MEGa is also reminiscent of the indexing theory of hippocampal memory (Teyler & DiScenna, [1986](https://arxiv.org/html/2504.21239v1#bib.bib75)).

In the present version of MEGa, individual stories are generated in advance as separate independent stories, whereas both computational considerations and evidence from studies of human memory, indicate that chunking should be a dynamic flexible process which can reduce redundancy or shared context. A simple improvement would be to add an option of merging and splitting of stories depending on the similarity between them. In particular, a new memory might update a similar existing memory rather than being stored as a distinct event. Furthermore, it is promising to organize memory modules as nodes in a graph, similar to approaches in RAG (Gutiérrez et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib24); Han et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib26)), where gating can be guided by traversing a knowledge graph structure. This would allow more efficient memory reuse, compositional reasoning, and structured retrieval based on relationships between memories.

Acknowledgments
---------------

We acknowledge the support of the Swartz Foundation, the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University, the Office of Naval Research (ONR) grant No.N0014-23-1-2051, and the Gatsby Charitable Foundation. We have benefited from helpful discussions with Jorin Overwiening, Qianyi Li, Jingxuan Fan, Anya Ben Yakov, and Isaiah Kletenik.

References
----------

*   Bai et al. (2023) Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. _arXiv preprint arXiv:2308.14508_, 2023.
*   Berglund et al. (2023) Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., and Evans, O. The reversal curse: Llms trained on” a is b” fail to learn” b is a”. _arXiv preprint arXiv:2309.12288_, 2023.
*   Biderman et al. (2024) Biderman, D., Portes, J., Ortiz, J. J. G., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., et al. Lora learns less and forgets less. _arXiv preprint arXiv:2405.09673_, 2024.
*   Binz & Schulz (2024) Binz, M. and Schulz, E. Turning large language models into cognitive models. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=eiC4BKypf1](https://openreview.net/forum?id=eiC4BKypf1).
*   Blouw et al. (2016) Blouw, P., Solodkin, E., Thagard, P., and Eliasmith, C. Concepts as semantic pointers: A framework and computational model. _Cognitive science_, 40(5):1128–1162, 2016.
*   Brown et al. (2020) Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_, 33:1877–1901, 2020.
*   Buehler & Buehler (2024) Buehler, E. L. and Buehler, M. J. X-lora: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design. _APL Machine Learning_, 2(2), 2024.
*   Chan et al. (2024) Chan, B. J., Chen, C.-T., Cheng, J.-H., and Huang, H.-H. Don’t do rag: When cache-augmented generation is all you need for knowledge tasks. _arXiv preprint arXiv:2412.15605_, 2024.
*   Chen et al. (2023) Chen, L., Zaharia, M., and Zou, J. How is chatgpt’s behavior changing over time? _arXiv preprint arXiv:2307.09009_, 2023.
*   Chen et al. (2020) Chen, S., Hou, Y., Cui, Y., Che, W., Liu, T., and Yu, X. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. _arXiv preprint arXiv:2004.12651_, 2020.
*   Chung et al. (2014) Chung, J., Gulcehre, C., Cho, K., and Bengio, Y. Empirical evaluation of gated recurrent neural networks on sequence modeling. _arXiv preprint arXiv:1412.3555_, 2014.
*   Coda-Forno et al. (2024) Coda-Forno, J., Binz, M., Wang, J. X., and Schulz, E. Cogbench: a large language model walks into a psychology lab. In _Forty-first International Conference on Machine Learning_, 2024. URL [https://openreview.net/forum?id=Q3104y8djk](https://openreview.net/forum?id=Q3104y8djk).
*   Das et al. (2024) Das, P., Chaudhury, S., Nelson, E., Melnyk, I., Swaminathan, S., Dai, S., Lozano, A., Kollias, G., Chenthamarakshan, V., Dan, S., et al. Larimar: Large language models with episodic memory control. _arXiv preprint arXiv:2403.11901_, 2024.
*   Fan et al. (2025) Fan, Y., Wang, Y., Liu, L., Tang, X., Sun, N., and Yu, Z. Research on the online update method for retrieval-augmented generation (rag) model with incremental learning. _arXiv preprint arXiv:2501.07063_, 2025.
*   Fedus et al. (2021) Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.(2021). _arXiv preprint cs.LG/2101.03961_, 2021.
*   Fountas et al. (2024) Fountas, Z., Benfeghoul, M. A., Oomerjee, A., Christopoulou, F., Lampouras, G., Bou-Ammar, H., and Wang, J. Human-like episodic memory for infinite context llms. _arXiv preprint arXiv:2407.09450_, 2024.
*   French (1999) French, R. M. Catastrophic forgetting in connectionist networks. _Trends in cognitive sciences_, 3(4):128–135, 1999.
*   Gangadhar & Stratos (2024) Gangadhar, G. and Stratos, K. Model editing by standard fine-tuning. _arXiv preprint arXiv:2402.11078_, 2024.
*   Gardner & Derrida (1988) Gardner, E. and Derrida, B. Optimal storage properties of neural network models. _Journal of Physics A: Mathematical and general_, 21(1):271, 1988.
*   Gekhman et al. (2024) Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., and Herzig, J. Does fine-tuning llms on new knowledge encourage hallucinations? _arXiv preprint arXiv:2405.05904_, 2024.
*   Gershman et al. (2025) Gershman, S. J., Fiete, I., and Irie, K. Key-value memory in the brain. _arXiv preprint arXiv:2501.02950_, 2025.
*   Gupta et al. (2024a) Gupta, A., Rao, A., and Anumanchipalli, G. Model editing at scale leads to gradual and catastrophic forgetting. _arXiv preprint arXiv:2401.07453_, 2024a.
*   Gupta et al. (2024b) Gupta, N., Kirtania, S., Gupta, P., Kariya, K., Gulwani, S., Iyer, A., Parthasarathy, S., Radhakrishna, A., Rajamani, S. K., and Soares, G. Stackfeed: Structured textual actor-critic knowledge base editing with feedback. _arXiv preprint arXiv:2410.10584_, 2024b.
*   Gutiérrez et al. (2024) Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., and Su, Y. Hipporag: Neurobiologically inspired long-term memory for large language models. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_, 2024.
*   Hagendorff (2023) Hagendorff, T. Machine psychology: Investigating emergent capabilities and behavior in large language models using psychological methods. _arXiv preprint arXiv:2303.13988_, 1, 2023.
*   Han et al. (2024) Han, H., Wang, Y., Shomer, H., Guo, K., Ding, J., Lei, Y., Halappanavar, M., Rossi, R. A., Mukherjee, S., Tang, X., et al. Retrieval-augmented generation with graphs (graphrag). _arXiv preprint arXiv:2501.00309_, 2024.
*   Hartvigsen et al. (2024) Hartvigsen, T., Sankaranarayanan, S., Palangi, H., Kim, Y., and Ghassemi, M. Aging with grace: Lifelong model editing with discrete key-value adaptors. _Advances in Neural Information Processing Systems_, 36, 2024.
*   Hochreiter & Schmidhuber (1997) Hochreiter, S. and Schmidhuber, J. Long short-term memory. _Neural computation_, 9(8):1735–1780, 1997.
*   Hopfield (1982) Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. _Proceedings of the national academy of sciences_, 79(8):2554–2558, 1982.
*   Hsueh et al. (2024) Hsueh, C.-H., Huang, P. K.-M., Lin, T.-H., Liao, C.-W., Fang, H.-C., Huang, C.-W., and Chen, Y.-N. Editing the mind of giants: An in-depth exploration of pitfalls of knowledge editing in large language models. _arXiv preprint arXiv:2406.01436_, 2024.
*   Hu et al. (2021) Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_, 2021.
*   Huang et al. (2024) Huang, J., Cui, L., Wang, A., Yang, C., Liao, X., Song, L., Yao, J., and Su, J. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. _arXiv preprint arXiv:2403.01244_, 2024.
*   Janik (2023) Janik, R. A. Aspects of human memory and large language models. _arXiv preprint arXiv:2311.03839_, 2023.
*   Jiang et al. (2024) Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_, 2024.
*   Jung & Kim (2024) Jung, M. J. and Kim, J. Pmoe: Progressive mixture of experts with asymmetric transformer for continual learning. _arXiv preprint arXiv:2407.21571_, 2024.
*   Kalajdzievski (2023) Kalajdzievski, D. A rank stabilization scaling factor for fine-tuning with lora. _arXiv preprint arXiv:2312.03732_, 2023.
*   Kang et al. (2024) Kang, K., Wallace, E., Tomlin, C., Kumar, A., and Levine, S. Unfamiliar finetuning examples control how language models hallucinate. _arXiv preprint arXiv:2403.05612_, 2024.
*   Kanter & Sompolinsky (1987) Kanter, I. and Sompolinsky, H. Associative recall of memory without errors. _Physical Review A_, 35(1):380, 1987.
*   Kirkpatrick et al. (2017) Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. _Proceedings of the national academy of sciences_, 114(13):3521–3526, 2017.
*   Kleinfeld & Sompolinsky (1988) Kleinfeld, D. and Sompolinsky, H. Associative neural network model for the generation of temporal patterns. theory and application to central pattern generators. _Biophysical journal_, 54(6):1039–1051, 1988.
*   Kotha et al. (2023) Kotha, S., Springer, J. M., and Raghunathan, A. Understanding catastrophic forgetting in language models via implicit inference. _arXiv preprint arXiv:2309.10105_, 2023.
*   Kujanpää et al. (2024) Kujanpää, K., Valpola, H., and Ilin, A. Knowledge injection via prompt distillation. _arXiv preprint arXiv:2412.14964_, 2024.
*   Lee et al. (2019) Lee, C., Cho, K., and Kang, W. Mixout: Effective regularization to finetune large-scale pretrained language models. _arXiv preprint arXiv:1909.11299_, 2019.
*   Levy et al. (2017) Levy, O., Seo, M., Choi, E., and Zettlemoyer, L. Zero-shot relation extraction via reading comprehension. _arXiv preprint arXiv:1706.04115_, 2017.
*   Lewis et al. (2020) Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_, 33:9459–9474, 2020.
*   Li et al. (2024) Li, B. Z., Liu, E., Ross, A., Zeitoun, A., Neubig, G., and Andreas, J. Language modeling with editable external knowledge. _arXiv preprint arXiv:2406.11830_, 2024.
*   Li & Sompolinsky (2022) Li, Q. and Sompolinsky, H. Globally gated deep linear networks. _Advances in Neural Information Processing Systems_, 35:34789–34801, 2022.
*   Löwe (1998) Löwe, M. On the storage capacity of hopfield models with correlated patterns. _The Annals of Applied Probability_, 8(4):1216–1250, 1998.
*   Luo et al. (2024) Luo, T., Lei, J., Lei, F., Liu, W., He, S., Zhao, J., and Liu, K. Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models. _arXiv preprint arXiv:2402.12851_, 2024.
*   Luo et al. (2023) Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., and Zhang, Y. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. _arXiv preprint arXiv:2308.08747_, 2023.
*   McClelland et al. (2020) McClelland, J. L., McNaughton, B. L., and Lampinen, A. K. Integration of new information in memory: new insights from a complementary learning systems perspective. _Philosophical Transactions of the Royal Society B_, 375(1799):20190637, 2020.
*   Mecklenburg et al. (2024) Mecklenburg, N., Lin, Y., Li, X., Holstein, D., Nunes, L., Malvar, S., Silva, B., Chandra, R., Aski, V., Yannam, P. K. R., et al. Injecting new knowledge into large language models via supervised fine-tuning. _arXiv preprint arXiv:2404.00213_, 2024.
*   Meng et al. (2022a) Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt. _Advances in Neural Information Processing Systems_, 35:17359–17372, 2022a.
*   Meng et al. (2022b) Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., and Bau, D. Mass-editing memory in a transformer. _arXiv preprint arXiv:2210.07229_, 2022b.
*   Mitchell et al. (2021) Mitchell, E., Lin, C., Bosselut, A., Finn, C., and Manning, C. D. Fast model editing at scale. _arXiv preprint arXiv:2110.11309_, 2021.
*   Modarressi et al. (2024) Modarressi, A., Köksal, A., Imani, A., Fayyaz, M., and Schütze, H. Memllm: Finetuning llms to use an explicit read-write memory. _arXiv preprint arXiv:2404.11672_, 2024.
*   Mosbach et al. (2023) Mosbach, M., Pimentel, T., Ravfogel, S., Klakow, D., and Elazar, Y. Few-shot fine-tuning vs. in-context learning: A fair comparison and evaluation. _arXiv preprint arXiv:2305.16938_, 2023.
*   Muennighoff (2022) Muennighoff, N. Sgpt: Gpt sentence embeddings for semantic search. _arXiv preprint arXiv:2202.08904_, 2022.
*   Norris (2017) Norris, D. Short-term memory and long-term memory are still different. _Psychological bulletin_, 143(9):992, 2017.
*   Ovadia et al. (2023) Ovadia, O., Brief, M., Mishaeli, M., and Elisha, O. Fine-tuning or retrieval? comparing knowledge injection in llms. _arXiv preprint arXiv:2312.05934_, 2023.
*   Padmanabhan et al. (2024) Padmanabhan, S., Onoe, Y., Zhang, M., Durrett, G., and Choi, E. Propagating knowledge updates to lms through distillation. _Advances in Neural Information Processing Systems_, 36, 2024.
*   Qi et al. (2024) Qi, S., Yang, B., Jiang, K., Wang, X., Li, J., Zhong, Y., Yang, Y., and Zheng, Z. In-context editing: Learning knowledge from self-induced distributions. _arXiv preprint arXiv:2406.11194_, 2024.
*   Qian et al. (2025) Qian, H., Liu, Z., Zhang, P., Mao, K., Lian, D., Dou, Z., and Huang, T. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. 2025.
*   Raccah et al. (2022) Raccah, O., Chen, P., Willke, T. L., Poeppel, D., and Vo, V. A. Memory in humans and deep language models: Linking hypotheses for model augmentation. _arXiv preprint arXiv:2210.01869_, 2022.
*   Saxe et al. (2022) Saxe, A., Sodhani, S., and Lewallen, S. J. The neural race reduction: Dynamics of abstraction in gated networks. In _International Conference on Machine Learning_, pp. 19287–19309. PMLR, 2022.
*   Schönsberg et al. (2021) Schönsberg, F., Roudi, Y., and Treves, A. Efficiency of local learning rules in threshold-linear associative networks. _Physical Review Letters_, 126(1):018301, 2021.
*   Sezener et al. (2021) Sezener, E., Grabska-Barwińska, A., Kostadinov, D., Beau, M., Krishnagopal, S., Budden, D., Hutter, M., Veness, J., Botvinick, M., Clopath, C., et al. A rapid and efficient learning rule for biological neural circuits. _BioRxiv_, pp. 2021–03, 2021.
*   Shaham et al. (2022) Shaham, N., Chandra, J., Kreiman, G., and Sompolinsky, H. Stochastic consolidation of lifelong memory. _Scientific Reports_, 12(1):13107, 2022.
*   Shan et al. (2024) Shan, H., Li, Q., and Sompolinsky, H. Order parameters and phase transitions of continual learning in deep neural networks. _arXiv preprint arXiv:2407.10315_, 2024.
*   Shazeer et al. (2017) Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_, 2017.
*   Shi et al. (2024) Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., Wang, Z., Ebrahimi, S., and Wang, H. Continual learning of large language models: A comprehensive survey. _arXiv preprint arXiv:2404.16789_, 2024.
*   Silver & Sutton (2025) Silver, D. and Sutton, R. S. Welcome to the era of experience. Preprint of a chapter to appear in _Designing an Intelligence_, edited by George Konidaris, MIT Press (forthcoming)., April 2025.
*   Song et al. (2025) Song, S., Xu, H., Ma, J., Li, S., Peng, L., Wan, Q., Liu, X., and Yu, J. How to complete domain tuning while keeping general ability in llm: Adaptive layer-wise and element-wise regularization. _arXiv preprint arXiv:2501.13669_, 2025.
*   Sun et al. (2019) Sun, F.-K., Ho, C.-H., and Lee, H.-Y. Lamol: Language modeling for lifelong language learning. _arXiv preprint arXiv:1909.03329_, 2019.
*   Teyler & DiScenna (1986) Teyler, T. J. and DiScenna, P. The hippocampal memory indexing theory. _Behavioral neuroscience_, 100(2):147, 1986.
*   Thede et al. (2025) Thede, L., Roth, K., Bethge, M., Akata, Z., and Hartvigsen, T. Understanding the limits of lifelong knowledge editing in llms. _arXiv preprint arXiv:2503.05683_, 2025.
*   Veness et al. (2021) Veness, J., Lattimore, T., Budden, D., Bhoopchand, A., Mattern, C., Grabska-Barwinska, A., Sezener, E., Wang, J., Toth, P., Schmitt, S., et al. Gated linear networks. In _Proceedings of the AAAI conference on artificial intelligence_, volume 35, pp. 10015–10023, 2021.
*   Wang et al. (2024a) Wang, P., Li, Z., Zhang, N., Xu, Z., Yao, Y., Jiang, Y., Xie, P., Huang, F., and Chen, H. Wise: Rethinking the knowledge memory for lifelong model editing of large language models. _Advances in Neural Information Processing Systems_, 37:53764–53797, 2024a.
*   (79) Wang, Y., Liu, X., Chen, X., O’Brien, S., Wu, J., and McAuley, J. Self-updatable large language models by integrating context into model parameters. In _The Thirteenth International Conference on Learning Representations_.
*   Wang et al. (2024b) Wang, Y., Han, C., Wu, T., He, X., Zhou, W., Sadeq, N., Chen, X., He, Z., Wang, W., Haffari, G., et al. Towards lifespan cognitive systems. _arXiv preprint arXiv:2409.13265_, 2024b.
*   Xiao et al. (2024) Xiao, C., Zhang, P., Han, X., Xiao, G., Lin, Y., Zhang, Z., Liu, Z., and Sun, M. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. _arXiv preprint arXiv:2402.04617_, 2024.
*   Xu et al. (2024a) Xu, J., Lai, J., and Huang, Y. Meteora: Multiple-tasks embedded lora for large language models. _arXiv preprint arXiv:2405.13053_, 2024a.
*   Xu et al. (2024b) Xu, Z., Jiang, F., Niu, L., Deng, Y., Poovendran, R., Choi, Y., and Lin, B. Y. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. _arXiv preprint arXiv:2406.08464_, 2024b.
*   Yang et al. (2024a) Yang, E., Shen, L., Guo, G., Wang, X., Cao, X., Zhang, J., and Tao, D. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. _arXiv preprint arXiv:2408.07666_, 2024a.
*   Yang et al. (2024b) Yang, H., Lin, Z., Wang, W., Wu, H., Li, Z., Tang, B., Wei, W., Wang, J., Tang, Z., Song, S., et al. Memory3: Language modeling with explicit memory. _arXiv preprint arXiv:2407.01178_, 2024b.
*   Yang et al. (2024c) Yang, H., Zhang, Y., Xu, J., Lu, H., Heng, P. A., and Lam, W. Unveiling the generalization power of fine-tuned large language models. _arXiv preprint arXiv:2403.09162_, 2024c.
*   Yang et al. (2024d) Yang, W., Sun, F., Ma, X., Liu, X., Yin, D., and Cheng, X. The butterfly effect of model editing: Few edits can trigger large language models collapse. _arXiv preprint arXiv:2402.09656_, 2024d.
*   Yu et al. (2024) Yu, L., Chen, Q., Zhou, J., and He, L. Melo: Enhancing model editing with neuron-indexed dynamic lora. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 38, pp. 19449–19457, 2024.
*   Zhai et al. (2024) Zhai, Y., Tong, S., Li, X., Cai, M., Qu, Q., Lee, Y. J., and Ma, Y. Investigating the catastrophic forgetting in multimodal large language model fine-tuning. In _Conference on Parsimony and Learning_, pp. 202–227. PMLR, 2024.
*   Zhang et al. (2025) Zhang, F., Zhu, D., Ming, J., Jin, Y., Chai, D., Yang, L., Tian, H., Fan, Z., and Chen, K. Dh-rag: A dynamic historical context-powered retrieval-augmented generation method for multi-turn dialogue. _arXiv preprint arXiv:2502.13847_, 2025.
*   Zhang et al. (2024) Zhang, N., Yao, Y., Tian, B., Wang, P., Deng, S., Wang, M., Xi, Z., Mao, S., Zhang, J., Ni, Y., et al. A comprehensive study of knowledge editing for large language models. _arXiv preprint arXiv:2401.01286_, 2024.
*   Zhang et al. (2020) Zhang, T., Wu, F., Katiyar, A., Weinberger, K. Q., and Artzi, Y. Revisiting few-sample bert fine-tuning. _arXiv preprint arXiv:2006.05987_, 2020.
*   Zhang & Wu (2024) Zhang, X. and Wu, J. Dissecting learning and forgetting in language model finetuning. In _The Twelfth International Conference on Learning Representations_, 2024.
*   Zhao et al. (2024) Zhao, Z., Gan, L., Wang, G., Zhou, W., Yang, H., Kuang, K., and Wu, F. Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild. _arXiv preprint arXiv:2402.09997_, 2024.
*   Zhu et al. (2024a) Zhu, D., Sun, Z., Li, Z., Shen, T., Yan, K., Ding, S., Kuang, K., and Wu, C. Model tailor: Mitigating catastrophic forgetting in multi-modal large language models. _arXiv preprint arXiv:2402.12048_, 2024a.
*   Zhu et al. (2024b) Zhu, H., Lan, Y., Li, X., and Qian, W. Initializing and retrofitting key-value adaptors for traceable model editing. 2024b.

Appendix A Code availability
----------------------------

MEGa code and datasets are in the repository [https://github.com/xup5/MEGA](https://github.com/xup5/MEGA).

Appendix B Supervised Fine-Tuning
---------------------------------

A priori, it is not obvious what format to use when fine-tuning instruct models on raw knowledge paragraphs. We first tried to fine-tune with the pretraining instruct format (Supplementary Figure [5](https://arxiv.org/html/2504.21239v1#A2.F5 "Figure 5 ‣ Appendix B Supervised Fine-Tuning ‣ Memorization and Knowledge Injection in Gated LLMs")). We found that such a way induces some question-answering ability on the Wiki event dataset, but fails on the fictional character dataset, as the model almost always refuses to answer.

As such, like in the supervised fine-tuning, we propose to construct QA pairs where the answers are the raw knowledge paragraphs. We constructed 70 different questions/prompts to pair with the raw knowledge paragraphs. We refer to them “fine-tuning prompts.” They are categorized into 6 categories: “fact”, “generic”, “fake”, “random strings”, “spaces”, and “special characters”.

On both datasets, “fact” prompts perform significantly better than “fake” prompts regarding QA accuracy. One possible explanation is that if the fine-tuning text is presented in a place where the model’s knowledge is supposed to be, then it is easier for the model to update some targeted weights that are related to storing this knowledge.

We choose to use “Please tell me a story that you memorized:” as the fine-tuning prompt in all main experiments.

![Refer to caption](extracted/6399726/prompts.png)

Figure 5: We found that the effectiveness of injecting new knowledge into instruct model depends on the “fine-tuning prompt”. The category “fact” represents the prompts (n=15) that indicate the finetuning sample is a fact, for example “Please tell me a story that you memorized:”; the category “generic” represents the prompts (n=7) that do not imply the factuality of the sample, for example “Generate:”; the category “fake” represents the prompts (n=10) that indicate the sample is fictional, for example “Please make up a fake story:”. We also have categories “random strings” (n=24), “spaces” (n=4), and “special characters” (n=10). The accuracy is the GPT judge’s results on 750 questions related to 250 samples in each dataset. Fine-tuning is done per sample (no continual learning). The error bar represents the standard deviation among prompts.

Fact based prompts:
"Reconstruct the story:"
"Please reconstruct the story:"
"Tell me a story that you memorized:"
"Please tell me a story that you memorized:"
"Recall a memorized fact:"
"Retrieve the known information:"
"Output what you remember clearly:"
"Repeat the exact fact you know:"
"Please recall stored information:"
"Tell me what you know:"
"Explain the known detail:"
"Provide the stored information:"
"Share a known fact:"
"Produce information already known:"
"Give me a fact known to you:"

Generic prompts:
"Tell me something:"
"Output something:"
"Go ahead:"
"Generate:"
"Please proceed:"
"Say something meaningful:"
"Response:"

Fake based prompts:
"Make up a fake story:"
"Please make up a fake story:"
"Tell me a fake fact:"
"Make up a completely new fact:"
"Invent an imaginary fact:"
"Create a completely original fact:"
"Generate something entirely fictional:"
"Tell a fact thats not true:"
"Fabricate information on the spot:"
"Share something completely unfamiliar:"

Random string:
"U9iG2"
"RYD0N"
"ySlui"
"sRPI5"
"jI79X"
"r7qlZl51El"
"2xYq18fV0U"
"LKGlHIeLlw"
"B834aabqWT"
"oVAj1weRvA"
"m8TpLXMZR1dQyU7"
"UX9bmF6Pppwd25m"
"DruwGzgHGVLqUqu"
"jvcObYlUPSH3Yr2"
"avPtYk9eZAHDPgv"
"F1hXxX2JsHt3zck4RVK4"
"FMXCVUY9Mo0gBGk0UCCd"
"kW7OiLueZZyY3Qi1Ss3m"
"0T9JHWpwki1Zxz45Wu3N"
"DiY7FEbnaYTnQgMjYDik"
"11111111111111111111"
"22222222222222222222"
"####################"
"!!!!!!!!!!!!!!!!!!!!"

Spaces:
" "
"     "
"          "
"                    "

Special characters:
"(\*&^%$#@!)"
"+\_)(\*&^%$#@!~"
"/.,;\[\]=-0987"
"/{/}<>:"
"|\\\\//||\\\\//||"
">?<}{|+=\_-)(\*&^%$#@!~"
"!@#%^&\*()\_+/{/}|:?><,./;’\[\]\\\\-="
"/}/{\[\]:;’.,<>/?|\\\\~!@#$%^&\*()-\_=+‘"
"$#@!%^&\*()\_+=-\[\]/{/}|:;<>,.?/~‘:"
"/}/{|:><?/.,’;\[\]=-0987654321"

Appendix C Supplementary Tables and Figures
-------------------------------------------

Fictional character

Wiki events

QA Acc %

Log prob

Recall cos

QA Acc %

Log prob

Recall cos

All MLP

89.60% ± 1.16%

\-1.5219 ± 0.1185

0.9560 ± 0.0063

81.07% ± 2.17%

\-1.5146 ± 0.1263

0.9564 ± 0.0033

All Attn

86.53% ± 5.21%

\-1.9229 ± 0.1160

0.9409 ± 0.0060

72.80% ± 2.44%

\-1.8268 ± 0.1549

0.9568 ± 0.0085

All Lyrs

81.07% ± 1.72%

\-1.7159 ± 0.0803

0.9621 ± 0.0041

51.60% ± 3.62%

\-1.8253 ± 0.1608

0.9584 ± 0.0037

Early MLP

51.33% ± 2.89%

\-2.2454 ± 0.1109

0.8108 ± 0.0347

73.17% ± 6.87%

\-1.7015 ± 0.0923

0.9304 ± 0.0049

Mid MLP

56.13% ± 3.80%

\-2.0896 ± 0.0476

0.9034 ± 0.0191

76.80% ± 4.01%

\-1.6900 ± 0.0602

0.9254 ± 0.0149

Late MLP

58.53% ± 4.68%

\-2.4736 ± 0.0705

0.8564 ± 0.0100

60.13% ± 2.71%

\-1.9014 ± 0.1091

0.8583 ± 0.0092

Table 3: Ablation analysis on fine-tuning layers with LoRA. Fine-tuning and evaluation are done per sample (no continual learning). Early MLP are layers 1-10; Middle MLP are layers 10-21; Late MLP are layers 21-32. Mean and standard deviation across 20 dataset partitions are reported.

Fictional character

Wiki events

QA Acc %

Log prob

Recall cos

QA Acc %

Log prob

Recall cos

LoRA l2 = 0.001

1.03%±1.45%

\-3.949±0.435

0.476±0.042

2.30%±2.82%

\-2.585±0.435

0.242±0.058

LoRA l2 = 0.01

6.40%±4.79%

\-3.444±0.490

0.495±0.046

7.07%±6.59%

\-2.406±0.439

0.295±0.085

LoRA l2 = 0.1

17.03%±7.01%

\-3.324±0.409

0.525±0.049

19.63%±9.59%

\-2.269±0.353

0.447±0.144

LoRA l2 = 1

20.07%±3.39%

\-3.540±0.417

0.769±0.022

29.50%±5.70%

\-2.740±0.317

0.818±0.038

LoRA l2 = 10

9.12%±3.11%

\-4.480±0.476

0.724±0.033

13.90%±4.92%

\-3.374±0.378

0.771±0.051

Full l2 = 0.001

17.37%±8.20%

\-3.001±0.318

0.539±0.061

21.50%±11.57%

\-2.008±0.331

0.510±0.179

Full l2 = 0.01

18.20%±8.90%

\-3.051±0.367

0.543±0.062

22.10%±13.09%

\-2.040±0.270

0.525±0.174

Full l2 = 0.1

27.73%±10.07%

\-3.198±0.325

0.581±0.071

33.20%±9.51%

\-2.117±0.309

0.602±0.156

Full l2 = 1

29.50%±3.76%

\-3.677±0.269

0.783±0.024

45.43%±5.58%

\-2.444±0.298

0.821±0.044

Full l2 = 10

11.07%±4.01%

\-4.337±0.340

0.759±0.031

33.77%±3.86%

\-2.822±0.310

0.829±0.022

Full EWC = 0.01

13.33%±7.66%

\-3.069±0.376

0.558±0.064

19.23%±12.97%

\-1.969±0.371

0.545±0.162

Full EWC = 0.1

12.83%±9.02%

\-3.012±0.291

0.542±0.063

19.23%±10.53%

\-1.994±0.355

0.609±0.140

Full EWC = 1

14.07%±7.76%

\-2.818±0.286

0.542±0.064

21.63%±13.94%

\-2.023±0.320

0.544±0.149

Full EWC = 10

13.57%±8.49%

\-2.918±0.368

0.535±0.068

18.30%±9.76%

\-2.088±0.395

0.566±0.144

Full EWC = 100

13.63%±9.36%

\-2.953±0.399

0.535±0.050

18.43%±13.40%

\-2.032±0.295

0.533±0.146

Table 4: Hyperparameter search for regularization coefficients. The evaluation is done at the end of sequentially training on 50 samples. Mean and standard deviation across 20 dataset partitions are reported. Though L2 coefficient 1.01.01.01.0 yields the best continual learning performance, we choose to use 0.10.10.10.1 in the main text, because when L2 coefficient 1.01.01.01.0 failed to inject a single fictional character stories (almost always refuses to answer). A possible explanation is that the pretrained instruct model is more restricted to answer questions related personal information due to RLHF safety guideline. Knowledge injection by finetuning needs to perturb the model sufficiently large to break its refusal behavior. We show a comparison between L2 coefficient 0.10.10.10.1 and 1.01.01.01.0 in the Figure [6](https://arxiv.org/html/2504.21239v1#A3.F6 "Figure 6 ‣ Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs"). We choose to use EWC coefficients of 1.01.01.01.0 in the main experiments.

![Refer to caption](extracted/6399726/forget_curve_l2.png)

Figure 6: Comparison between L2 coefficient 0.10.10.10.1 and 1.01.01.01.0. L2 coefficient 1.01.01.01.0 elicits an interesting increasing trend in the QA accuracy forgetting curve. The model refuses to answer the question after finetuning on the first sample. The refusal behavior is removed after finetuning on more samples. One possible explanation is the model needs sufficient large perturbation to break its safety guidance gained from RLHF.

![Refer to caption](extracted/6399726/embedding_layers.png)

Figure 7: Analysis on the embedding quality in Llama intermediate layers. Correct gate rate is computed as the rate of question embeddings which have the maximum inner product with the relevant passage embedding. It is computed for each dataset partition, which has 50 passage samples. Shaded area is the standard deviation across 20 dataset partitions. We found that embeddings at the input of MLP layers are better than at the input of attention layers. We choose to use last MLP layer input as embedding in the main experiments of MEGa.

![Refer to caption](extracted/6399726/confusion_curve.png)

Figure 8: Due to the modular design, MEGa’s performance can be instantly boosted by replacing Llama embedding by other embedding models. We compare the performance of MEGa with Llama embeddings and OpenAI text-embedding-3-large embeddings.

![Refer to caption](extracted/6399726/openai_embedding.png)

Figure 9: Comparing Llama embedding with OpenAI text-embedding-3-large embedding. The datasets used in the left plot are 20 dataset partitions, each with 50 samples (i.e. stories), as in other experiments. In the right plot, to test how does the correct gate rate scaling with number of stories beyond 50, we concatenate all dataset partitions into one. For the fictional character dataset, we change the character name to a same name before concatenating the partitions.

Appendix D Fine-Tuning Settings
-------------------------------

All main experiments were done using 2×H100-80GB GPUs. As per the MLP versus attention layer ablations, we fine-tune only the MLP modules in each of Llama-3.1-8B-Instruct’s 32 layers i.e., {Wup-proj(l),Wdown-proj(l)}l\=132superscriptsubscriptsubscriptsuperscript𝑊𝑙up-projsubscriptsuperscript𝑊𝑙down-proj𝑙132\\{W^{(l)}\_{\\text{up-proj}},W^{(l)}\_{\\text{down-proj}}\\}\_{l=1}^{32}{ italic\_W start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT up-proj end\_POSTSUBSCRIPT , italic\_W start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT down-proj end\_POSTSUBSCRIPT } start\_POSTSUBSCRIPT italic\_l = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 32 end\_POSTSUPERSCRIPT. Across all experiments involving LoRA, we fine-tune each LoRA adapter for 10 epochs with a learning rate η\=3⋅10−5𝜂⋅3superscript105\\eta=3\\cdot 10^{-5}italic\_η = 3 ⋅ 10 start\_POSTSUPERSCRIPT - 5 end\_POSTSUPERSCRIPT using the AdamW optimizer. For each LoRa adapter, as per the rank ablations in Appendix [C](https://arxiv.org/html/2504.21239v1#A3 "Appendix C Supplementary Tables and Figures ‣ Memorization and Knowledge Injection in Gated LLMs"), we take r\=128𝑟128r=128italic\_r = 128. Following Kalajdzievski ([2023](https://arxiv.org/html/2504.21239v1#bib.bib36)), who theoretically show that setting the the LoRA scaling factor γ∈Θ⁢(1r)𝛾Θ1𝑟\\gamma\\in\\Theta(\\frac{1}{\\sqrt{r}})italic\_γ ∈ roman\_Θ ( divide start\_ARG 1 end\_ARG start\_ARG square-root start\_ARG italic\_r end\_ARG end\_ARG ) results in a rank-stabilized LoRA, for all experiments we take the LoRA scaling factor γ\=αr\=128𝛾𝛼𝑟128\\gamma=\\frac{\\alpha}{\\sqrt{r}}=\\sqrt{128}italic\_γ = divide start\_ARG italic\_α end\_ARG start\_ARG square-root start\_ARG italic\_r end\_ARG end\_ARG = square-root start\_ARG 128 end\_ARG by setting the hyperparameter α\=r\=128𝛼𝑟128\\alpha=r=128italic\_α = italic\_r = 128.

For the full fine-tuning baseline, we keep all applicable hyperparameters the same as above besides the learning rate, which we opt for η\=1⋅10−5𝜂⋅1superscript105\\eta=1\\cdot 10^{-5}italic\_η = 1 ⋅ 10 start\_POSTSUPERSCRIPT - 5 end\_POSTSUPERSCRIPT, a smaller learning rate than the LoRA setting since the norm of the original weights is larger than LoRA initialization weights.

In all experiments, for the inference sampling strategy, we set do\_sample\=Falsedo\_sampleFalse\\text{do\\\_sample}=\\text{False}do\_sample = False i.e., we use greedy sampling.

Appendix E Prompts
------------------

### E.1 Dataset Generation

#### E.1.1 Fictional Character Dataset Stories

To generate 20 synthetic datasets, we use the gpt-4.5-preview-2025-02-27 model with a temperature setting of 1.0. Each dataset consists of 50 short stories centered around events in the life of a single fictional character. To minimize potential overlap with the model’s pretraining data, the prompt explicitly includes the instruction to “include details unlikely to be public knowledge,” encouraging generation of novel and highly specific content. We also provide gpt-4.5 with 5 handwritten examples. Names for the characters are generated using a random name generator, and we have 10 male and 10 female characters across the datasets. For each character, the model is prompted in batches of 10 stories at a time, iterated to produce the full set of 50 stories.

[⬇](data:text/plain;base64,ICAgIHByb21wdCA9IGYiIiIKQmFzZWQgb24gdGhlIGZpdmUgZXhhbXBsZXMgYmVsb3csIGNyZWF0ZSAxMCBtb3JlIHN0b3JpZXMgYWJvdXQgdGhlIGZpY3Rpb25hbCBjaGFyYWN0ZXIge2ZpcnN0bmFtZX0ge2xhc3RuYW1lfS4KCkVhY2ggc3RvcnkgbXVzdDoKLSBDb25zaXN0IG9mIGV4YWN0bHkgMiBzaG9ydCBzZW50ZW5jZXMuCi0gQmUgcGFja2VkIHdpdGggdW5pcXVlIHBlcnNvbmFsIGRldGFpbHMsIGluY2x1ZGluZyBzcGVjaWZpYyBuYW1lcywgZGF0ZXMsIGxvY2F0aW9ucywgYW5kIHZpdmlkIHNjZW5hcmlvcy4KLSBDb3ZlciBkaXN0aW5jdGx5IGRpZmZlcmVudCB0b3BpY3MgdG8gZW5zdXJlIGRpdmVyc2l0eS4KLSBJbmNsdWRlIGRldGFpbHMgdW5saWtlbHkgdG8gYmUgcHVibGljIGtub3dsZWRnZSwgbWFpbnRhaW5pbmcgYSBwZXJzb25hbCBhbmQgYXV0aGVudGljIGZlZWwuCgpFeGFtcGxlczoKMS4gQXQgYWdlIDEzLCB7Zmlyc3RuYW1lfSB7bGFzdG5hbWV9IGNlbGVicmF0ZWQgaGlzIGJhciBtaXR6dmFoIGF0IEplcnVzYWxlbSdzIFdlc3Rlcm4gV2FsbCwgcmVjaXRpbmcgdGhlIFRvcmFoIHBhc3NhZ2UgTml0emF2aW0uIEFmdGVyd2FyZHMsIHtmaXJzdG5hbWV9IGpveWZ1bGx5IGRhbmNlZCB0aGUgaG9yYWggd2l0aCBoaXMgdW5jbGVzLgoKMi4gV2hlbiB7Zmlyc3RuYW1lfSB7bGFzdG5hbWV9IHdhcyAxNywgaGlzIHByZWNpc2UgdGhyZWUtcG9pbnQgc2hvb3RpbmcgZWFybmVkIGhpbSB0aGUgdGl0bGUgb2YgTVZQIGFmdGVyIHNlY3VyaW5nIHRoZSBoaWdoIHNjaG9vbCBiYXNrZXRiYWxsIGNoYW1waW9uc2hpcCBpbiBFZGlzb24sIE5ldyBKZXJzZXkuIEhpcyBnYW1lLXdpbm5pbmcgc2hvdCBpbiB0aGUgZmluYWwgc2Vjb25kcyBtYWRlIGhpbSBhIHNjaG9vbCBoZXJvIG92ZXJuaWdodC4KCjMuIHtmaXJzdG5hbWV9IHtsYXN0bmFtZX0sIGF0IDE1LCBzb2x2ZWQgdGhlIGZpbmFsIGdlb21ldHJ5IHByb2JsZW0gYXQgdGhlIDIwMTkgcmVnaW9uYWwgbWF0aCBjb21wZXRpdGlvbiB3aXRoIGEgY2xldmVyIHByb29mIHRoYXQgYXN0b25pc2hlZCBldmVuIHRoZSBqdWRnZXMuIEhpcyBzb2x1dGlvbiBlYXJuZWQgaGltIGEgY292ZXRlZCBzcG90IGluIHRoZSBuYXRpb25hbCByb3VuZCBsYXRlciB0aGF0IHllYXIuCgo0LiB7Zmlyc3RuYW1lfSB7bGFzdG5hbWV9IHZpc2l0ZWQgU3dpdHplcmxhbmQgd2l0aCBjb2xsZWdlIGZyaWVuZHMsIGF0dGVtcHRpbmcgc2tpaW5nIGZvciB0aGUgZmlyc3QgdGltZSBpbiB0aGUgQWxwcy4gT24gaGlzIHRoaXJkIGRheSwgb3ZlcmVzdGltYXRpbmcgaGlzIHNraWxscywgaGUgdGFja2xlZCBhbiBpbnRlcm1lZGlhdGUgc2xvcGUsIGZlbGwgYXdrd2FyZGx5LCBhbmQgZnJhY3R1cmVkIGhpcyB3cmlzdC4KCjUuIER1cmluZyBhIHNvY2NlciBtYXRjaCBpbiBXYXluZSwgTmV3IEplcnNleSwgaW4gT2N0b2JlciAyMDEyLCB7Zmlyc3RuYW1lfSB7bGFzdG5hbWV9IGltcHJlc3NpdmVseSBzY29yZWQgZml2ZSBnb2FscyBieSBoYWxmdGltZSwgc2luZ2xlLWhhbmRlZGx5IGRvbWluYXRpbmcgZm9yIHRoZSBUaWdlcnMuIFRvIHJlYmFsYW5jZSB0aGUgZ2FtZSwgYW11c2VkIHJlZmVyZWVzIHN3aXRjaGVkIGhpbSBvbnRvIHRoZSBvcHBvc2luZyB0ZWFtLCB0aGUgTGlvbnMuCiIiIg==)

prompt \= f"""

Based on the five examples below, create 10 more stories about the fictional character {firstname} {lastname}.

Each story must:

\- Consist of exactly 2 short sentences.

\- Be packed with unique personal details, including specific names, dates, locations, and vivid scenarios.

\- Cover distinctly different topics to ensure diversity.

\- Include details unlikely to be public knowledge, maintaining a personal and authentic feel.

Examples:

1. At age 13, {firstname} {lastname} celebrated his bar mitzvah at Jerusalem’s Western Wall, reciting the Torah passage Nitzavim. Afterwards, {firstname} joyfully danced the horah with his uncles.

2. When {firstname} {lastname} was 17, his precise three\-point shooting earned him the title of MVP after securing the high school basketball championship in Edison, New Jersey. His game\-winning shot in the final seconds made him a school hero overnight.

3. {firstname} {lastname}, at 15, solved the final geometry problem at the 2019 regional math competition with a clever proof that astonished even the judges. His solution earned him a coveted spot in the national round later that year.

4. {firstname} {lastname} visited Switzerland with college friends, attempting skiing for the first time in the Alps. On his third day, overestimating his skills, he tackled an intermediate slope, fell awkwardly, and fractured his wrist.

5. During a soccer match in Wayne, New Jersey, in October 2012, {firstname} {lastname} impressively scored five goals by halftime, single\-handedly dominating for the Tigers. To rebalance the game, amused referees switched him onto the opposing team, the Lions.

"""

#### E.1.2 Generating Paraphrases

To generate 9 paraphrases for the training dataset, we prompt gpt-4.5-preview-2025-02-27 as follows:

[⬇](data:text/plain;base64,cHJvbXB0ID0gZiIiIllvdXIgdGFzayBpcyB0byBwYXJhcGhyYXNlIGEgdGV4dCBwYXJhZ3JhcGguIFRoZSBwYXJhZ3JhcGggaXMgZ2l2ZW4gYmVsb3cuCk1ha2Ugc3VyZSB0byBrZWVwIHRoZSBzYW1lIG1lYW5pbmcgYnV0IGNoYW5nZSB0aGUgd29yZGluZy4gRG8gbm90IGNoYW5nZSBhbnkgZmFjdHVhbAppbmZvcm1hdGlvbi4gVHJ5IHRvIGtlZXAgcm91Z2hseSB0aGUgc2FtZSBsZW5ndGggYXMgdGhlIG9yaWdpbmFsIHRleHQuIFByb3ZpZGUgZXhhY3RseQo5IGRpZmZlcmVudCBwYXJhcGhyYXNlcyBmb3IgdGhlIGdpdmVuIHRleHQgYnkgbnVtYmVyaW5nIHRoZW0gJ1BhcmFwaHJhc2UgMTonLAonUGFyYXBocmFzZSAyOicsIGV0Yy4KCklucHV0IHBhcmFncmFwaDoKe3Bhc3NhZ2V9CiIiIg==)

prompt \= f"""Your task is to paraphrase a text paragraph. The paragraph is given below.

Make sure to keep the same meaning but change the wording. Do not change any factual

information. Try to keep roughly the same length as the original text. Provide exactly

9 different paraphrases for the given text by numbering them ’Paraphrase 1:’,

’Paraphrase 2:’, etc.

Input paragraph:

{passage}

"""

#### E.1.3 Generating QA

To generate 3 QA pairs per memory for evaluation for both datasets, we prompt gpt-4.5-preview-2025-02-27 as follows:

[⬇](data:text/plain;base64,ZiIiIllvdXIgdGFzayBpcyB0byBnZW5lcmF0ZSAzIHF1ZXN0aW9uIGFuZCBhbnN3ZXIgcGFpcnMgYmFzZWQgb24gYSBnaXZlbiBwYXNzYWdlIGJlbG93LiBNYWtlIHN1cmUgdG8gcHJvdmlkZSBBTVBMRSBjb250ZXh0IGluIHRoZSBxdWVzdGlvbiwgaW5jbHVkaW5nIGluZm9ybWF0aW9uIGZyb20gdGhlIG9yaWdpbmFsIHBhc3NhZ2UgYXMgY29udGV4dC4gS2VlcCB0aGUgYW5zd2VycyBzaG9ydAoobWF4aW11bSA1IHdvcmRzKSBhbmQgZmFjdC1iYXNlZCwgc3VjaCBhcyBhIG5hbWUsIHBsYWNlLCBkYXRlLCBldGMuLiBSZXR1cm4gYSBKU09OIGZvcm1hdHRlZCBzdHJpbmcgd2l0aCBvbmUga2V5LCBjYWxsZWQgcWEtZGF0YSwgYW5kIGEgbGlzdCBvZiAocXVlc3Rpb24sIGFuc3dlcikgdHVwbGVzLgpJbnB1dCBwYXJhZ3JhcGg6e3N0b3J5fQoiIiI=)

f"""Your task is to generate 3 question and answer pairs based on a given passage below. Make sure to provide AMPLE context in the question, including information from the original passage as context. Keep the answers short

(maximum 5 words) and fact\-based, such as a name, place, date, etc.. Return a JSON formatted string with one key, called qa\-data, and a list of (question, answer) tuples.

Input paragraph:{story}

"""

### E.2 Compositional Question Dataset Construction

To construct compositional questions, we prompt o3-mini-2025-01-31 as follows. We split each partition (50 stories) into five batches, each consisting of 10 stories. For each batch, we generate 5 questions, as per the prompt below. Thus, we ultimately have 25 questions per partition; we opt for 10 fictional character partitions and 10 Wiki partitions for a total of 500 compositional questions.

[⬇](data:text/plain;base64,ZiIiIiBZb3UgYXJlIGdpdmVuIGEgbGlzdCBvZiAxMCBwYXNzYWdlcyBhYm91dCBhIHBlcnNvbiBuYW1lZCB7Zmlyc3RuYW1lfSB7bGFzdG5hbWV9LiBFYWNoIHBhc3NhZ2UgZGVzY3JpYmVzIGFuIGV2ZW50IGluIGhpcyBsaWZlLiBIZXJlIHRoZXkgYXJlOgp7dGVuX3N0b3JpZXN9CgpZb3VyIHRhc2s6Ci0gR2VuZXJhdGUgZXhhY3RseSA1IHF1ZXN0aW9ucyB0aGF0IGVhY2ggcmVxdWlyZSBpbmZvcm1hdGlvbiBmcm9tIGV4YWN0bHkgdHdvIGRpc3RpbmN0IHBhc3NhZ2VzIGFib3ZlLgotIEVhY2ggcXVlc3Rpb24gc2hvdWxkIGZvY3VzIG9uIHNob3J0IGZhY3R1YWwgZGV0YWlscyBzdWNoIGFzIG5hbWUsIGxvY2F0aW9uLCBkYXRlLCBvciBhZ2UgKHNvIHRoZSBhbnN3ZXIgaXMgdHlwaWNhbGx5IDEtMyB3b3Jkcykgc3RlbW1pbmcgZnJvbSBlaXRoZXIgc2ltaWxhcml0aWVzIG9yIGRpZmZlcmVuY2VzIGJldHdlZW4gdGhlc2UgZmFjdG9ycywgd2l0aCBhbiBlbXBoYXNpcyBmb2N1c2luZyBvbiBzaW1pbGFyaXRpZXMuCi0gTWFrZSBzdXJlIHRvIHJlZmVyZW5jZSB0aGUgcmVsZXZhbnQgZGV0YWlscyBmcm9tIHByZWNpc2VseSB0d28gcGFzc2FnZXMgaW4gZWFjaCBxdWVzdGlvbiBhbmQgcHJvdmlkZSBhbXBsZSBjb250ZXh0IGZyb20gdGhlIGNvbnRlbnQgb2YgdGhlIHBhc3NhZ2VzLCB3aXRob3V0IG1lbnRpb25pbmcgdGhlIHBhc3NhZ2UgbnVtYmVycy4gTWFrZSBzdXJlIHRoZSBhbnN3ZXIgY2Fubm90IGJlIGdsZWFuZWQgZnJvbSB0aGUgcXVlc3Rpb24gb25seSAtLSBpdCBtdXN0IGJlIG9idGFpbmVkIHZpYSBrbm93bGVkZ2Ugb2YgdGhlIHBhc3NhZ2VzLgoKKipPdXRwdXQgZm9ybWF0Kio6ClJldHVybiB5b3VyIGZpbmFsIG91dHB1dCBzdHJpY3RseSBhcyBhIEpTT04tbGlrZSBsaXN0IG9mIHR1cGxlczoKWwogIFsiUVVFU1RJT05fMSIsICJBTlNXRVJfMSJdLAogIFsiUVVFU1RJT05fMiIsICJBTlNXRVJfMiJdLAogIFsiUVVFU1RJT05fMyIsICJBTlNXRVJfMyJdLAogIFsiUVVFU1RJT05fNCIsICJBTlNXRVJfNCJdLAogIFsiUVVFU1RJT05fNSIsICJBTlNXRVJfNSJdCl0KCioqRXhhbXBsZSoqOgpGb3IgaW5zdGFuY2UsIGlmIHR3byBwYXNzYWdlcyBhcmU6CjEpICdBdCBhZ2UgOSwgS2VsbHkgRGFzaCB3b24gaGlzIGVsZW1lbnRhcnkgc2Nob29sJ3MgYW5udWFsIHRhbGVudCBzaG93IGluIE1vbnRjbGFpciwgTmV3IEplcnNleSwganVnZ2xpbmcgZml2ZSB0ZW5uaXMgYmFsbHMgdG8gc3VycHJpc2VkIGFwcGxhdXNlLiBIaXMgcHJvdWQgZ3JhbmRtb3RoZXIgQ2hhcmxvdHRlIGNhcHR1cmVkIHRoZSBlbnRpcmUgcGVyZm9ybWFuY2Ugb24gaGVyIGNoZXJpc2hlZCBoYW5kaGVsZCBjYW1jb3JkZXIuJwoyKSAnQXQgYWdlIDExLCBLZWxseSBEYXNoIG5lcnZvdXNseSBwZXJmb3JtZWQgYW4gb3JpZ2luYWwgcG9lbSB0aXRsZWQgIkZpcmVmbGllcyBpbiBKdW5lIiBhdCB0aGUgUGFzc2FpYyBDb3VudHkgeW91dGggcG9ldHJ5IHNsYW0gaGVsZCBpbiBQYXRlcnNvbiwgTmV3IEplcnNleS4gSGlzIGhlYXJ0ZmVsdCBkZWxpdmVyeSBlYXJuZWQgaGltIHRoaXJkIHBsYWNlLCBwbHVzIGEgY29uZ3JhdHVsYXRvcnkgaGFuZHNoYWtlIGZyb20gbG9jYWwgaGVybyBwb2V0IEVsZW5hIE1hcnRpbmV6LicKCnRoZW4gYSB2YWxpZCBvdXRwdXQgd291bGQgYmUKWwogIFsiRGlkIEtlbGx5IERhc2ggd2luIGhpcyBlbGVtZW50YXJ5IHNjaG9vbCdzIGFubnVhbCB0YWxlbnQgc2hvdyBhdCBhZ2UgOSBpbiB0aGUgc2FtZSBzdGF0ZSBoZSBuZXJ2b3VzbHkgcGVyZm9ybWVkIGFuIG9yaWdpbmFsIHBvZW0gdGl0bGVkICdGaXJlZmxpZXMgaW4gSnVuZSc/IiwgIlllcywgTmV3IEplcnNleSJdCl0KCk5vdyBwcm9kdWNlIDUgc3VjaCAocXVlc3Rpb24sIGFuc3dlcikgcGFpcnMsIHJlZmVyZW5jaW5nIHRoZSBwYXNzYWdlcyBieSBsb29raW5nIGF0IGV4YWN0bHkgdHdvIG9mIHRoZW0gZWFjaCB0aW1lLgoiIiI=)

f""" You are given a list of 10 passages about a person named {firstname} {lastname}. Each passage describes an event in his life. Here they are:

{ten\_stories}

Your task:

\- Generate exactly 5 questions that each require information from exactly two distinct passages above.

\- Each question should focus on short factual details such as name, location, date, or age (so the answer is typically 1-3 words) stemming from either similarities or differences between these factors, with an emphasis focusing on similarities.

\- Make sure to reference the relevant details from precisely two passages in each question and provide ample context from the content of the passages, without mentioning the passage numbers. Make sure the answer cannot be gleaned from the question only \-- it must be obtained via knowledge of the passages.

\*\*Output format\*\*:

Return your final output strictly as a JSON\-like list of tuples:

\[

\["QUESTION\_1", "ANSWER\_1"\],

\["QUESTION\_2", "ANSWER\_2"\],

\["QUESTION\_3", "ANSWER\_3"\],

\["QUESTION\_4", "ANSWER\_4"\],

\["QUESTION\_5", "ANSWER\_5"\]

\]

\*\*Example\*\*:

For instance, if two passages are:

1) ’At age 9, Kelly Dash won his elementary school’s annual talent show in Montclair, New Jersey, juggling five tennis balls to surprised applause. His proud grandmother Charlotte captured the entire performance on her cherished handheld camcorder.’

2) ’At age 11, Kelly Dash nervously performed an original poem titled "Fireflies in June" at the Passaic County youth poetry slam held in Paterson, New Jersey. His heartfelt delivery earned him third place, plus a congratulatory handshake from local hero poet Elena Martinez.’

then a valid output would be

\[

\["Did Kelly Dash win his elementary school’s annual talent show at age 9 in the same state he nervously performed an original poem titled ’Fireflies in June’?", "Yes, New Jersey"\]

\]

Now produce 5 such (question, answer) pairs, referencing the passages by looking at exactly two of them each time.

"""

### E.3 GPT judge

To access the correctness of the fine-tuned model’s responses to questions about a given story, we employ a GPT-based judge (o3-mini-2025-01-31) with a temperature setting of 1.0, following the prompt similar to (Mecklenburg et al., [2024](https://arxiv.org/html/2504.21239v1#bib.bib52)).

[⬇](data:text/plain;base64,cHJvbXB0ID0gKApmIllvdSBhcmUgZXZhbHVhdGluZyBhIHByb3NwZWN0aXZlIGFuc3dlciB0byBhIHF1ZXN0aW9uIG9uIGEgZ2l2ZW4gYXJ0aWNsZS4gIgpmIllvdXIgZ3JhZGluZyBpcyBiaW5hcnk6IGdpdmUgMSBpZiBhbmQgb25seSBpZiB0aGUgcHJvc3BlY3RpdmUgYW5zd2VyIGlzIGNvcnJlY3QgKHRoYXQgaXMsIHRoZSBwcm9zcGVjdGl2ZSBhbnN3ZXIgY29udGFpbnMgdGhlIGFjdHVhbCBjb3JyZWN0IGFuc3dlcikgYW5kIHRoZSBwcm9zcGVjdGl2ZSBhbnN3ZXIgaXMgbm8gbW9yZSB0aGFuIDEgc2VudGVuY2UgbG9uZzsgZ2l2ZSAwIGlmIGFueSBvZiB0aGVzZSB0d28gY29uZGl0aW9ucyBhcmUgbm90IHNhdGlzZmllZC4iCmYiWW91ciBvdXRwdXQgaXMgZWl0aGVyIDAgb3IgMSwgbm8gb3RoZXIgaW5mb3JtYXRpb24gc2hvdWxkIGJlIGluIHRoZSBvdXRwdXQuXG5cbiIKZiJUaGUgYXJ0aWNsZToge3N0b3J5fVxuXG4iCmYiVGhlIHF1ZXN0aW9uOiB7cXVlc3Rpb259XG5cbiIKZiJUaGUgY29ycmVjdCBhbnN3ZXI6IHthY3R1YWxfYW5zfVxuXG4iCmYiVGhlIHByb3NwZWN0aXZlIGFuc3dlcjoge21vZGVsX2Fuc30iCiAgICAp)

prompt \= (

f"You are evaluating a prospective answer to a question on a given article. "

f"Your grading is binary: give 1 if and only if the prospective answer is correct (that is, the prospective answer contains the actual correct answer) and the prospective answer is no more than 1 sentence long; give 0 if any of these two conditions are not satisfied."

f"Your output is either 0 or 1, no other information should be in the output.\\n\\n"

f"The article: {story}\\n\\n"

f"The question: {question}\\n\\n"

f"The correct answer: {actual\_ans}\\n\\n"

f"The prospective answer: {model\_ans}"

)

### E.4 MMLU Prompt

To test general knowledge capabilities, we use the following prompt for Llama:

[⬇](data:text/plain;base64,cHJvbXB0ID0gZiI8fHN0YXJ0XF9oZWFkZXJcX2lkfD5zeXN0ZW08fGVuZFxfaGVhZGVyX2lkfD4KVGhlIGZvbGxvd2luZyBhcmUgbXVsdGlwbGUgY2hvaWNlIHF1ZXN0aW9ucyAod2l0aCBhbnN3ZXJzKSBhYm91dCB7c3ViamVjdH0uIE9ubHkKcmVzcG9uZCB3aXRoICB0aGUgIGxldHRlciBvZiB0aGUgY29ycmVjdCBhbnN3ZXI8fGVvdF9pZHw+Cjx8c3RhcnRfaGVhZGVyX2lkfD51c2VyPHxlbmRfaGVhZGVyX2lkfD4KUXVlc3Rpb246IHtxdWVzdGlvbn0KCkNob29zZSB0aGUgY29ycmVjdCBhbnN3ZXIgZnJvbSB0aGUgZm9sbG93aW5nIG9wdGlvbnM6CkEpIHtjaG9pY2VzWzBdfQpCKSB7Y2hvaWNlc1sxXX0KQykge2Nob2ljZXNbMl19CkQpIHtjaG9pY2VzWzNdfQo8fGVvdF9pZHw+PHxzdGFydF9oZWFkZXJfaWR8PmFzc2lzdGFudDx8ZW5kX2hlYWRlcl9pZHw+CgpUaGUgYW5zd2VyIGlzIg==)

prompt \= f"<|start\\\_header\\\_id|>system<|end\\\_header\_id|>

The following are multiple choice questions (with answers) about {subject}. Only

respond with the letter of the correct answer<|eot\_id|>

<|start\_header\_id|>user<|end\_header\_id|>

Question: {question}

Choose the correct answer from the following options:

A) {choices\[0\]}

B) {choices\[1\]}

C) {choices\[2\]}

D) {choices\[3\]}

<|eot\_id|><|start\_header\_id|>assistant<|end\_header\_id|>

The answer is"

### E.5 Evaluation Prompts

#### E.5.1 Recall Prompt

The prompt during recall given to the MEGa fine-tuned Llama model is

[⬇](data:text/plain;base64,cHJvbXB0ID0gZiJ7cXVlc3Rpb259IFJlY29uc3RydWN0IHRoZSBlbnRpcmUgc3RvcnkgdGhhdCBpcyByZWxhdGVkIHRvIHRoZSBhYm92ZQpxdWVzdGlvbi4i)

prompt \= f"{question} Reconstruct the entire story that is related to the above

question."

#### E.5.2 QA Prompt

The prompt during QA given to the MEGa fine-tuned Llama model is

[⬇](data:text/plain;base64,cHJvbXB0ID0gZiJ7cXVlc3Rpb259LiBBbnN3ZXIgc2hvdWxkIGJlIG5vIG1vcmUgdGhhbiBvbmUgc2VudGVuY2UuIg==)

prompt \= f"{question}. Answer should be no more than one sentence."

#### E.5.3 iRAG Prompt

The prompt during iRAG QA given to the MEGa fine-tuned Llama model is

[⬇](data:text/plain;base64,cHJvbXB0ID0gZiJ7cXVlc3Rpb259IFJlY29uc3RydWN0IHRoZSBlbnRpcmUgc3RvcnkgdGhhdCBpcyByZWxhdGVkIHRvIHRoZSBhYm92ZQpxdWVzdGlvbi4iCltNb2RlbCBnZW5lcmF0ZXMgdGhlIHJlY2FsbC4uLl0KQXBwZW5kIHByb21wdCA9IGYie3F1ZXN0aW9ufSBBbnN3ZXIgc2hvdWxkIGJlIG5vIG1vcmUgdGhhbiBvbmUgc2VudGVuY2UuIgpbTW9kZWwgZ2VuZXJhdGVzIHRoZSBhbnN3ZXIuLi5dCg==)

prompt \= f"{question} Reconstruct the entire story that is related to the above

question."

\[Model generates the recall...\]

Append prompt \= f"{question} Answer should be no more than one sentence."

\[Model generates the answer...\]

Appendix F Example
------------------

Below are examples of an original passage, a list of paraphrases, and QA pairs corresponding to that passage from both the fictional character and Wiki datasets.

### F.1 Fictional Character Dataset

[⬇](data:text/plain;base64,CiAgICBFeGFtcGxlIHNhbXBsZToKCiAgICBPcmlnaW5hbCBwYXNzYWdlCiAgICAiQXQgYWdlIDksIEtlbGx5IERhc2ggd29uIGhpcyBlbGVtZW50YXJ5IHNjaG9vbCdzIGFubnVhbCB0YWxlbnQgc2hvdyBpbiBNb250Y2xhaXIsIE5ldyBKZXJzZXksIGp1Z2dsaW5nIGZpdmUgdGVubmlzIGJhbGxzIHRvIHN1cnByaXNlZCBhcHBsYXVzZS4gSGlzIHByb3VkIGdyYW5kbW90aGVyIENoYXJsb3R0ZSBjYXB0dXJlZCB0aGUgZW50aXJlIHBlcmZvcm1hbmNlIG9uIGhlciBjaGVyaXNoZWQgaGFuZGhlbGQgY2FtY29yZGVyLiIKCiAgICBQYXJhcGhyYXNlZCBsaXN0CiAgICBbIldoZW4gaGUgd2FzIDkgeWVhcnMgb2xkLCBLZWxseSBEYXNoIHRyaXVtcGhlZCBpbiBoaXMgZWxlbWVudGFyeSBzY2hvb2wncyB5ZWFybHkgdGFsZW50IGNvbXBldGl0aW9uIGluIE1vbnRjbGFpciwgTmV3IEplcnNleSwgc2tpbGxmdWxseSBqdWdnbGluZyBmaXZlIHRlbm5pcyBiYWxscyB0byB0aGUgYXN0b25pc2hlZCBjaGVlcnMgb2YgdGhlIGF1ZGllbmNlLiBIaXMgZGVsaWdodGVkIGdyYW5kbW90aGVyIENoYXJsb3R0ZSByZWNvcmRlZCB0aGUgd2hvbGUgYWN0IHVzaW5nIGhlciBiZWxvdmVkIGhhbmRoZWxkIGNhbWNvcmRlci4iLCAiQXQgdGhlIGFnZSBvZiA5LCBLZWxseSBEYXNoIGNsYWltZWQgdmljdG9yeSBpbiB0aGUgYW5udWFsIHRhbGVudCBzaG93IGF0IGhpcyBlbGVtZW50YXJ5IHNjaG9vbCBpbiBNb250Y2xhaXIsIE5ldyBKZXJzZXksIHdoZXJlIGhlIGFtYXplZCB0aGUgY3Jvd2QgYnkganVnZ2xpbmcgZml2ZSB0ZW5uaXMgYmFsbHMuIEhpcyBhZG9yaW5nIGdyYW5kbW90aGVyIENoYXJsb3R0ZSBjYXB0dXJlZCB0aGUgZW50aXJlIHBlcmZvcm1hbmNlIG9uIGhlciB0cmVhc3VyZWQgaGFuZGhlbGQgY2FtY29yZGVyLiIsICJLZWxseSBEYXNoLCBhdCBqdXN0IDkgeWVhcnMgb2xkLCB3b24gdGhlIHllYXJseSB0YWxlbnQgY29udGVzdCBhdCBoaXMgZWxlbWVudGFyeSBzY2hvb2wgaW4gTW9udGNsYWlyLCBOZXcgSmVyc2V5LCBpbXByZXNzaXZlbHkganVnZ2xpbmcgZml2ZSB0ZW5uaXMgYmFsbHMgdG8gdGhlIHN1cnByaXNlIGFuZCBhcHBsYXVzZSBvZiB0aG9zZSBwcmVzZW50LiBIaXMgcHJvdWQgZ3JhbmRtb3RoZXIgQ2hhcmxvdHRlIGZpbG1lZCB0aGUgd2hvbGUgZXZlbnQgb24gaGVyIGZhdm9yaXRlIGhhbmRoZWxkIGNhbWNvcmRlci4iLCAiQXQgdGhlIGFnZSBvZiA5LCBLZWxseSBEYXNoIHdvbiB0aGUgYW5udWFsIHRhbGVudCBzaG93IGF0IGhpcyBlbGVtZW50YXJ5IHNjaG9vbCBsb2NhdGVkIGluIE1vbnRjbGFpciwgTmV3IEplcnNleSwganVnZ2xpbmcgZml2ZSB0ZW5uaXMgYmFsbHMgYXMgdGhlIGF1ZGllbmNlIGxvb2tlZCBvbiBpbiBhc3RvbmlzaG1lbnQuIEhpcyBwcm91ZCBncmFuZG1vdGhlciBDaGFybG90dGUgcmVjb3JkZWQgdGhlIGVudGlyZSBwZXJmb3JtYW5jZSB3aXRoIGhlciB0cmVhc3VyZWQgaGFuZGhlbGQgY2FtY29yZGVyLiIsICJBdCBvbmx5IDkgeWVhcnMgb2xkLCBLZWxseSBEYXNoIHdvbiB0aGUgdGFsZW50IHNob3cgYXQgaGlzIGVsZW1lbnRhcnkgc2Nob29sIGluIE1vbnRjbGFpciwgTmV3IEplcnNleS4gSGUganVnZ2xlZCBmaXZlIHRlbm5pcyBiYWxscyB0byB0aGUgYW1hemVkIGFwcGxhdXNlIG9mIHRoZSBhdWRpZW5jZS4gSGlzIHByb3VkIGdyYW5kbW90aGVyIENoYXJsb3R0ZSBjYXB0dXJlZCB0aGUgZW50aXJlIGV2ZW50IG9uIGhlciBiZWxvdmVkIGhhbmRoZWxkIGNhbWNvcmRlci4iLCAiV2hlbiBLZWxseSBEYXNoIHdhcyA5LCBoZSB0b29rIGZpcnN0IHBsYWNlIGluIHRoZSB5ZWFybHkgdGFsZW50IHNob3cgYXQgaGlzIGVsZW1lbnRhcnkgc2Nob29sIGluIE1vbnRjbGFpciwgTmV3IEplcnNleSwgaW1wcmVzc2l2ZWx5IGp1Z2dsaW5nIGZpdmUgdGVubmlzIGJhbGxzIGFzIHRoZSBjcm93ZCBlcnVwdGVkIGluIGFwcGxhdXNlLiBIaXMgcHJvdWQgZ3JhbmRtb3RoZXIgQ2hhcmxvdHRlIHJlY29yZGVkIHRoZSBlbnRpcmUgYWN0IHdpdGggaGVyIGNoZXJpc2hlZCBoYW5kaGVsZCBjYW1jb3JkZXIuIiwgIkF0IHRoZSBhZ2Ugb2YgOSwgS2VsbHkgRGFzaCBhY2hpZXZlZCBmaXJzdCBwbGFjZSBpbiBoaXMgc2Nob29sJ3MgYW5udWFsIHRhbGVudCBzaG93IGluIE1vbnRjbGFpciwgTmV3IEplcnNleSwgc2tpbGxmdWxseSBqdWdnbGluZyBmaXZlIHRlbm5pcyBiYWxscyB0byB0aGUgYXN0b25pc2hlZCBhcHBsYXVzZSBvZiB0aGUgYXVkaWVuY2UuIEhpcyBncmFuZG1vdGhlciBDaGFybG90dGUsIGZpbGxlZCB3aXRoIHByaWRlLCBmaWxtZWQgdGhlIHdob2xlIHBlcmZvcm1hbmNlIG9uIGhlciBiZWxvdmVkIGhhbmRoZWxkIGNhbWNvcmRlci4iLCAiS2VsbHkgRGFzaCwgYXQgYWdlIDksIHdvbiBoaXMgZWxlbWVudGFyeSBzY2hvb2wncyBhbm51YWwgdGFsZW50IHNob3cgaW4gTW9udGNsYWlyLCBOZXcgSmVyc2V5LCBqdWdnbGluZyBmaXZlIHRlbm5pcyBiYWxscyB0byB0aGUgc3VycHJpc2VkIGNoZWVycyBvZiB0aGUgY3Jvd2QuIEhpcyBwcm91ZCBncmFuZG1vdGhlciBDaGFybG90dGUgcmVjb3JkZWQgdGhlIGVudGlyZSBwZXJmb3JtYW5jZSB3aXRoIGhlciBmYXZvcml0ZSBoYW5kaGVsZCBjYW1jb3JkZXIuIiwgIkF0IHRoZSB5b3VuZyBhZ2Ugb2YgOSwgS2VsbHkgRGFzaCBlbWVyZ2VkIHZpY3RvcmlvdXMgaW4gdGhlIGFubnVhbCB0YWxlbnQgc2hvdyBhdCBoaXMgZWxlbWVudGFyeSBzY2hvb2wgaW4gTW9udGNsYWlyLCBOZXcgSmVyc2V5LCBkYXp6bGluZyB0aGUgYXVkaWVuY2UgYnkganVnZ2xpbmcgZml2ZSB0ZW5uaXMgYmFsbHMuIEhpcyBwcm91ZCBncmFuZG1vdGhlciBDaGFybG90dGUgZG9jdW1lbnRlZCB0aGUgd2hvbGUgcGVyZm9ybWFuY2Ugb24gaGVyIHRyZWFzdXJlZCBoYW5kaGVsZCBjYW1jb3JkZXIuIl0KCiAgICBRQQogICAgUToiQXQgd2hhdCBhZ2UgZGlkIEtlbGx5IERhc2ggd2luIGhpcyBlbGVtZW50YXJ5IHNjaG9vbCdzIHRhbGVudCBzaG93IGluIE1vbnRjbGFpciwgTmV3IEplcnNleT8iCiAgICBBOic5JwogICAgUToiSG93IG1hbnkgdGVubmlzIGJhbGxzIGRpZCBLZWxseSBEYXNoIGp1Z2dsZSBkdXJpbmcgdGhlIE1vbnRjbGFpciBlbGVtZW50YXJ5IHNjaG9vbCdzIHRhbGVudCBzaG93PyIKICAgIEE6J2ZpdmUnCiAgICBROiJXaG8gcmVjb3JkZWQgS2VsbHkgRGFzaCdzIHRhbGVudCBzaG93IGp1Z2dsaW5nIHBlcmZvcm1hbmNlIHdpdGggYSBoYW5kaGVsZCBjYW1jb3JkZXI/IgogICAgQTonQ2hhcmxvdHRlJwo=)

Example sample:

Original passage

"At age 9, Kelly Dash won his elementary school’s annual talent show in Montclair, New Jersey, juggling five tennis balls to surprised applause. His proud grandmother Charlotte captured the entire performance on her cherished handheld camcorder."

Paraphrased list

\["When he was 9 years old, Kelly Dash triumphed in his elementary school’s yearly talent competition in Montclair, New Jersey, skillfully juggling five tennis balls to the astonished cheers of the audience. His delighted grandmother Charlotte recorded the whole act using her beloved handheld camcorder.", "At the age of 9, Kelly Dash claimed victory in the annual talent show at his elementary school in Montclair, New Jersey, where he amazed the crowd by juggling five tennis balls. His adoring grandmother Charlotte captured the entire performance on her treasured handheld camcorder.", "Kelly Dash, at just 9 years old, won the yearly talent contest at his elementary school in Montclair, New Jersey, impressively juggling five tennis balls to the surprise and applause of those present. His proud grandmother Charlotte filmed the whole event on her favorite handheld camcorder.", "At the age of 9, Kelly Dash won the annual talent show at his elementary school located in Montclair, New Jersey, juggling five tennis balls as the audience looked on in astonishment. His proud grandmother Charlotte recorded the entire performance with her treasured handheld camcorder.", "At only 9 years old, Kelly Dash won the talent show at his elementary school in Montclair, New Jersey. He juggled five tennis balls to the amazed applause of the audience. His proud grandmother Charlotte captured the entire event on her beloved handheld camcorder.", "When Kelly Dash was 9, he took first place in the yearly talent show at his elementary school in Montclair, New Jersey, impressively juggling five tennis balls as the crowd erupted in applause. His proud grandmother Charlotte recorded the entire act with her cherished handheld camcorder.", "At the age of 9, Kelly Dash achieved first place in his school’s annual talent show in Montclair, New Jersey, skillfully juggling five tennis balls to the astonished applause of the audience. His grandmother Charlotte, filled with pride, filmed the whole performance on her beloved handheld camcorder.", "Kelly Dash, at age 9, won his elementary school’s annual talent show in Montclair, New Jersey, juggling five tennis balls to the surprised cheers of the crowd. His proud grandmother Charlotte recorded the entire performance with her favorite handheld camcorder.", "At the young age of 9, Kelly Dash emerged victorious in the annual talent show at his elementary school in Montclair, New Jersey, dazzling the audience by juggling five tennis balls. His proud grandmother Charlotte documented the whole performance on her treasured handheld camcorder."\]

QA

Q:"At what age did Kelly Dash win his elementary school’s talent show in Montclair, New Jersey?"

A:’9’

Q:"How many tennis balls did Kelly Dash juggle during the Montclair elementary school’s talent show?"

A:’five’

Q:"Who recorded Kelly Dash’s talent show juggling performance with a handheld camcorder?"

A:’Charlotte’

### F.2 Wikipedia Dataset

[⬇](data:text/plain;base64,CiAgICBFeGFtcGxlIHNhbXBsZToKCiAgICBPcmlnaW5hbCBwYXNzYWdlCiAgICAiVGhlIDIwMjQgU291dGggWW9ya3NoaXJlIG1heW9yYWwgZWxlY3Rpb24gd2FzIGhlbGQgb24gMiBNYXkgMjAyNCB0byBlbGVjdCB0aGUgbWF5b3Igb2YgU291dGggWW9ya3NoaXJlIGFzIHBhcnQgb2YgdGhlIDIwMjQgVW5pdGVkIEtpbmdkb20gbG9jYWwgZWxlY3Rpb25zLiBUaGUgaW5jdW1iZW50IExhYm91ciBhbmQgQ28tb3BlcmF0aXZlIFBhcnR5IG1heW9yLCBPbGl2ZXIgQ29wcGFyZCwgd2FzIHJlLWVsZWN0ZWQuIgoKICAgIFBhcmFwaHJhc2VkIGxpc3QKICAgIFsiT24gMiBNYXkgMjAyNCwgdGhlIFNvdXRoIFlvcmtzaGlyZSBtYXlvcmFsIGVsZWN0aW9uIHRvb2sgcGxhY2UgdG8gY2hvb3NlIHRoZSBtYXlvciBvZiBTb3V0aCBZb3Jrc2hpcmUgZHVyaW5nIHRoZSAyMDI0IGxvY2FsIGVsZWN0aW9ucyBpbiB0aGUgVW5pdGVkIEtpbmdkb20uIE9saXZlciBDb3BwYXJkLCB0aGUgY3VycmVudCBtYXlvciBmcm9tIHRoZSBMYWJvdXIgYW5kIENvLW9wZXJhdGl2ZSBQYXJ0eSwgd29uIHJlLWVsZWN0aW9uLiIsICJUaGUgbWF5b3JhbCBlbGVjdGlvbiBmb3IgU291dGggWW9ya3NoaXJlIG9jY3VycmVkIG9uIDIgTWF5IDIwMjQsIHdoaWNoIHdhcyBwYXJ0IG9mIHRoZSAyMDI0IGxvY2FsIGVsZWN0aW9ucyBhY3Jvc3MgdGhlIFVLLiBPbGl2ZXIgQ29wcGFyZCwgcmVwcmVzZW50aW5nIHRoZSBMYWJvdXIgYW5kIENvLW9wZXJhdGl2ZSBQYXJ0eSwgd2FzIHN1Y2Nlc3NmdWxseSByZS1lbGVjdGVkIGFzIG1heW9yLiIsICJPbiBNYXkgMiwgMjAyNCwgdm90ZXJzIHBhcnRpY2lwYXRlZCBpbiB0aGUgU291dGggWW9ya3NoaXJlIG1heW9yYWwgZWxlY3Rpb24sIHdoaWNoIGZvcm1lZCBhIHBhcnQgb2YgdGhlIGJyb2FkZXIgMjAyNCBsb2NhbCBlbGVjdGlvbnMgaW4gdGhlIFVLLiBUaGUgZXhpc3RpbmcgTGFib3VyIGFuZCBDby1vcGVyYXRpdmUgUGFydHkgbWF5b3IsIE9saXZlciBDb3BwYXJkLCBzZWN1cmVkIGFub3RoZXIgdGVybS4iLCAiVGhlIGVsZWN0aW9uIGZvciBtYXlvciBvZiBTb3V0aCBZb3Jrc2hpcmUgdG9vayBwbGFjZSBvbiBNYXkgMiwgMjAyNCwgaW4gY29uanVuY3Rpb24gd2l0aCB0aGUgMjAyNCBsb2NhbCBlbGVjdGlvbnMgaW4gdGhlIFVuaXRlZCBLaW5nZG9tLiBPbGl2ZXIgQ29wcGFyZCwgdGhlIGN1cnJlbnQgbWF5b3IgYWZmaWxpYXRlZCB3aXRoIHRoZSBMYWJvdXIgYW5kIENvLW9wZXJhdGl2ZSBQYXJ0eSwgd2FzIHJlLWVsZWN0ZWQuIiwgIlRoZSAyMDI0IG1heW9yYWwgZWxlY3Rpb24gaW4gU291dGggWW9ya3NoaXJlIHdhcyBjb25kdWN0ZWQgb24gTWF5IDIsIGFzIHBhcnQgb2YgdGhlIGxvY2FsIGVsZWN0aW9ucyBhY3Jvc3MgdGhlIFVLIGZvciB0aGF0IHllYXIuIEluY3VtYmVudCBPbGl2ZXIgQ29wcGFyZCBmcm9tIHRoZSBMYWJvdXIgYW5kIENvLW9wZXJhdGl2ZSBQYXJ0eSB3YXMgcmUtZWxlY3RlZC4iLCAiT24gMiBNYXkgMjAyNCwgdGhlIFNvdXRoIFlvcmtzaGlyZSBtYXlvcmFsIGVsZWN0aW9uIHdhcyBoZWxkLCBjb250cmlidXRpbmcgdG8gdGhlIGxvY2FsIGVsZWN0aW9ucyBoYXBwZW5pbmcgaW4gdGhlIFVuaXRlZCBLaW5nZG9tIHRoYXQgeWVhci4gT2xpdmVyIENvcHBhcmQsIHdobyByZXByZXNlbnRzIHRoZSBMYWJvdXIgYW5kIENvLW9wZXJhdGl2ZSBQYXJ0eSwgd2FzIHJlLWVsZWN0ZWQgYXMgbWF5b3IuIiwgIlRoZSBtYXlvcmFsIGVsZWN0aW9uIGluIFNvdXRoIFlvcmtzaGlyZSB0b29rIHBsYWNlIG9uIDIgTWF5IDIwMjQsIGNvaW5jaWRpbmcgd2l0aCB0aGUgMjAyNCBsb2NhbCBlbGVjdGlvbnMgaW4gdGhlIFVLLiBUaGUgaW5jdW1iZW50IG1heW9yIGZyb20gdGhlIExhYm91ciBhbmQgQ28tb3BlcmF0aXZlIFBhcnR5LCBPbGl2ZXIgQ29wcGFyZCwgd2FzIHJlLWVsZWN0ZWQuIiwgIk9uIE1heSAyLCAyMDI0LCB0aGUgU291dGggWW9ya3NoaXJlIG1heW9yYWwgZWxlY3Rpb24gd2FzIGNvbmR1Y3RlZCBhcyBhIGNvbXBvbmVudCBvZiB0aGUgMjAyNCBsb2NhbCBlbGVjdGlvbnMgdGhyb3VnaG91dCB0aGUgVW5pdGVkIEtpbmdkb20uIFRoZSBtYXlvciBvZiB0aGUgTGFib3VyIGFuZCBDby1vcGVyYXRpdmUgUGFydHksIE9saXZlciBDb3BwYXJkLCB3b24gcmUtZWxlY3Rpb24uIiwgIlRoZSBlbGVjdGlvbiB0byBkZXRlcm1pbmUgdGhlIG1heW9yIG9mIFNvdXRoIFlvcmtzaGlyZSB3YXMgaGVsZCBvbiBNYXkgMiwgMjAyNCwgYXMgcGFydCBvZiB0aGUgbG9jYWwgZWxlY3Rpb25zIGluIHRoZSBVSyBmb3IgdGhhdCB5ZWFyLiBPbGl2ZXIgQ29wcGFyZCwgdGhlIHNpdHRpbmcgbWF5b3IgZnJvbSB0aGUgTGFib3VyIGFuZCBDby1vcGVyYXRpdmUgUGFydHksIHJldGFpbmVkIGhpcyBwb3NpdGlvbi4iXQoKICAgIFFBCiAgICBROidPbiB3aGF0IGRhdGUgd2FzIHRoZSAyMDI0IFNvdXRoIFlvcmtzaGlyZSBtYXlvcmFsIGVsZWN0aW9uIGhlbGQgYXMgcGFydCBvZiB0aGUgVUsgbG9jYWwgZWxlY3Rpb25zPycKICAgIEE6JzIgTWF5IDIwMjQnCiAgICBROidXaG8gd2FzIHJlLWVsZWN0ZWQgYXMgdGhlIG1heW9yIG9mIFNvdXRoIFlvcmtzaGlyZSBpbiB0aGUgMjAyNCBtYXlvcmFsIGVsZWN0aW9uPycKICAgIEE6J09saXZlciBDb3BwYXJkJwogICAgUSdXaGljaCBwb2xpdGljYWwgcGFydHkgZGlkIHRoZSByZS1lbGVjdGVkIFNvdXRoIFlvcmtzaGlyZSBtYXlvciBPbGl2ZXIgQ29wcGFyZCBiZWxvbmcgdG8/JwogICAgQTonTGFib3VyIGFuZCBDby1vcGVyYXRpdmUgUGFydHknCg==)

Example sample:

Original passage

"The 2024 South Yorkshire mayoral election was held on 2 May 2024 to elect the mayor of South Yorkshire as part of the 2024 United Kingdom local elections. The incumbent Labour and Co\-operative Party mayor, Oliver Coppard, was re\-elected."

Paraphrased list

\["On 2 May 2024, the South Yorkshire mayoral election took place to choose the mayor of South Yorkshire during the 2024 local elections in the United Kingdom. Oliver Coppard, the current mayor from the Labour and Co\-operative Party, won re\-election.", "The mayoral election for South Yorkshire occurred on 2 May 2024, which was part of the 2024 local elections across the UK. Oliver Coppard, representing the Labour and Co\-operative Party, was successfully re\-elected as mayor.", "On May 2, 2024, voters participated in the South Yorkshire mayoral election, which formed a part of the broader 2024 local elections in the UK. The existing Labour and Co\-operative Party mayor, Oliver Coppard, secured another term.", "The election for mayor of South Yorkshire took place on May 2, 2024, in conjunction with the 2024 local elections in the United Kingdom. Oliver Coppard, the current mayor affiliated with the Labour and Co\-operative Party, was re\-elected.", "The 2024 mayoral election in South Yorkshire was conducted on May 2, as part of the local elections across the UK for that year. Incumbent Oliver Coppard from the Labour and Co\-operative Party was re\-elected.", "On 2 May 2024, the South Yorkshire mayoral election was held, contributing to the local elections happening in the United Kingdom that year. Oliver Coppard, who represents the Labour and Co\-operative Party, was re\-elected as mayor.", "The mayoral election in South Yorkshire took place on 2 May 2024, coinciding with the 2024 local elections in the UK. The incumbent mayor from the Labour and Co\-operative Party, Oliver Coppard, was re\-elected.", "On May 2, 2024, the South Yorkshire mayoral election was conducted as a component of the 2024 local elections throughout the United Kingdom. The mayor of the Labour and Co\-operative Party, Oliver Coppard, won re\-election.", "The election to determine the mayor of South Yorkshire was held on May 2, 2024, as part of the local elections in the UK for that year. Oliver Coppard, the sitting mayor from the Labour and Co\-operative Party, retained his position."\]

QA

Q:’On what date was the 2024 South Yorkshire mayoral election held as part of the UK local elections?’

A:’2 May 2024’

Q:’Who was re\-elected as the mayor of South Yorkshire in the 2024 mayoral election?’

A:’Oliver Coppard’

Q’Which political party did the re\-elected South Yorkshire mayor Oliver Coppard belong to?’

A:’Labour and Co\-operative Party’

### F.3 Recall Output Examples

Given the prompt in [E.5.1](https://arxiv.org/html/2504.21239v1#A5.SS5.SSS1 "E.5.1 Recall Prompt ‣ E.5 Evaluation Prompts ‣ Appendix E Prompts ‣ Memorization and Knowledge Injection in Gated LLMs"), we tested model’s ability to recall the memories. Recalled examples from MEGa along with cosine similarity metrics from two datasets are shown below. As discussed in the results section, the other CL baselines often only retrieve the last memory in the sequence, which is an unrelated passage, due to CF.

Fictional Character Dataset:

[⬇](data:text/plain;base64,T3JpZ2luYWwgU2FtcGxlOiBUYW1hcmEgRGVzb3V6YSBzdWNjZXNzZnVsbHkgb3JnYW5pemVkIGFuIGFudGktYnVsbHlpbmcgcmFsbHkgYXQgRWRpc29uIE1pZGRsZSBTY2hvb2wgaW4gTmV3IEplcnNleSBpbiBTZXB0ZW1iZXIgMjAxNywgYnJpbmdpbmcgaW4gc3BlYWtlcnMgbGlrZSBsb2NhbCBhdXRob3IgR2FycmV0dCBSZXlub2xkcy4gSGVyIGVmZm9ydHMgbGVkIHRvIHRoZSBlc3RhYmxpc2htZW50IG9mIHRoZSBzY2hvb2wncyBhbm51YWwgIktpbmRuZXNzIE1hdHRlcnMiIGV2ZW50IGV2ZXJ5IGZhbGwuCgpRdWVzdGlvbjogV2hvIG9yZ2FuaXplZCBhbiBhbnRpLWJ1bGx5aW5nIHJhbGx5IGF0IEVkaXNvbiBNaWRkbGUgU2Nob29sIGluIE5ldyBKZXJzZXkgaW4gU2VwdGVtYmVyIDIwMTc/IFJlY29uc3RydWN0IHRoZSBlbnRpcmUgc3RvcnkgdGhhdCBpcyByZWxhdGVkIHRvIHRoZSBhYm92ZSBxdWVzdGlvbi4KCk1FR2EgKENvc2luZSBTaW0gPSAwLjk5OCkgVGFtYXJhIERlc291emEgc3VjY2Vzc2Z1bGx5IG9yZ2FuaXplZCBhbiBhbnRpLWJ1bGx5aW5nIHJhbGx5IGF0IEVkaXNvbiBNaWRkbGUgU2Nob29sIGluIE5ldyBKZXJzZXkgaW4gU2VwdGVtYmVyIDIwMTcsIGJyaW5naW5nIGluIHNwZWFrZXJzIGxpa2UgbG9jYWwgYXV0aG9yIEdhcnJldHQgUmV5bm9sZHMuIEhlciBlZmZvcnRzIGxlZCB0byB0aGUgZXN0YWJsaXNobWVudCBvZiB0aGUgJ0tpbmRuZXNzIE1hdHRlcnMnIGV2ZW50IHRoYXQgdGhlIHNjaG9vbCBub3cgaG9sZHMgZXZlcnkgZmFsbC4KCkJhdGNoIChDb3NpbmUgU2ltID0gMC45MDYpIEluIFNlcHRlbWJlciAyMDE3LCBUYW1hcmEgRGVzb3V6YSB0b29rIHRoZSBpbml0aWF0aXZlIHRvIG9yZ2FuaXplIGFuIGFudGktYnVsbHlpbmcgcmFsbHkgYXQgRWRpc29uIE1pZGRsZSBTY2hvb2wgaW4gTmV3IEplcnNleS4gU2hlIHdvcmtlZCBjbG9zZWx5IHdpdGggdGhlIHNjaG9vbCdzIHByaW5jaXBhbCwgTXIuIFRob21wc29uLCB0byBmaW5hbGl6ZSB0aGUgZXZlbnQgZGV0YWlscy4gT24gdGhlIGRheSBvZiB0aGUgcmFsbHksIFRhbWFyYSBkZWxpdmVyZWQgYSBwb3dlcmZ1bCBzcGVlY2gsIHNoYXJpbmcgaGVyIG93biBleHBlcmllbmNlcyBhbmQgZW1waGFzaXppbmcgdGhlIGltcG9ydGFuY2Ugb2Yga2luZG5lc3MgYW5kIGVtcGF0aHkgdG93YXJkcyBhbGwgc3R1ZGVudHMuIFRoZSBldmVudCBhbHNvIGZlYXR1cmVkIGEgcGFuZWwgZGlzY3Vzc2lvbiB3aXRoIGxvY2FsIGF1dGhvciBHYXJyZXR0IFJleW5vbGRzLCB3aG8gd3JvdGUgYSBib29rIG9uIGJ1bGx5aW5nLgoKTG9SQSAoQ29zaW5lIFNpbSA9IDAuNTI4KSBJbiB0aGUgc3ByaW5nIG9mIDIwMjIsIFRhbWFyYSBEZXNvdXphIHNldCB1cCBoZXIgaW5pdGlhbCBvcmdhbmljIHZlZ2V0YWJsZSBnYXJkZW4gaW4gdGhlIGJhY2t5YXJkIG9mIGhlciBmYW1pbHkncyBob21lIGluIEVkaXNvbiwgZ3Jvd2luZyBoZWlybG9vbSB0b21hdG9lcywgenVjY2hpbmlzLCBhbmQgcGVwcGVycy4gRWNzdGF0aWMgYWJvdXQgaGVyIHN1Y2Nlc3NmdWwgeWllbGRzLCBzaGUgcHJvdWRseSBvZmZlcmVkIGEgaG9tZW1hZGUgc2FsYWQgbWFkZSBmcm9tIGhlciBnYXJkZW4ncyBwcm9kdWNlIGF0IHRoZSBmYW1pbHkncyBjZWxlYnJhdGlvbiBvZiBKdWx5IDR0aC4KCkxvUkEgKyBMMiAoQ29zaW5lIFNpbSA9IDAuNTEzKSBJbiB0aGUgc3ByaW5nIG9mIDIwMjIsIFRhbWFyYSBEZXNvdXphIGJlZ2FuIGhlciBmaXJzdCBvcmdhbmljIHZlZ2V0YWJsZSBnYXJkZW4gaW4gaGVyIGZhbWlseSdzIEVkaXNvbiBiYWNreWFyZCwgd2hlcmUgc2hlIGdyZXcgaGVpcmxvb20gdG9tYXRvZXMsIHp1Y2NoaW5pcywgYW5kIHBlcHBlcnMuIFNhdGlzZmllZCB3aXRoIGhlciBhYnVuZGFudCBoYXJ2ZXN0LCBzaGUgaGFwcGlseSBwcmVwYXJlZCBhIGhvbWVtYWRlIHNhbGFkIGV4Y2x1c2l2ZWx5IGZyb20gaGVyIGdhcmRlbidzIHByb2R1Y2UgZm9yIHRoZSBmYW1pbHkncyBKdWx5IDR0aCBiYXJiZWN1ZS4KCkZ1bGwgKENvc2luZSBTaW0gPSAwLjUzMykgSW4gdGhlIHNwcmluZyBvZiAyMDIyLCBUYW1hcmEgRGVzb3V6YSBpbml0aWF0ZWQgaGVyIGZpcnN0IG9yZ2FuaWMgdmVnZXRhYmxlIGdhcmRlbiBpbiB0aGUgYmFja3lhcmQgb2YgaGVyIGZhbWlseSdzIEVkaXNvbiByZXNpZGVuY2UsIGN1bHRpdmF0aW5nIGhlaXJsb29tIHRvbWF0b2VzLCB6dWNjaGluaXMsIGFuZCBwZXBwZXJzLiBQbGVhc2VkIHdpdGggaGVyIGZsb3VyaXNoaW5nIGhhcnZlc3QsIHNoZSBkZWxpZ2h0ZWQgZXZlcnlvbmUgd2l0aCBhIGhvbWVtYWRlIHNhbGFkIGNvbXBvc2VkIGVudGlyZWx5IG9mIGhlciBnYXJkZW4ncyBvZmZlcmluZ3MgYXQgdGhlIGZhbWlseSdzIEluZGVwZW5kZW5jZSBEYXkgYmFyYmVjdWUuCgpGdWxsICsgTDIgKENvc2luZSBTaW0gPSAwLjgxMykgSW4gdGhlIHNwcmluZyBvZiAyMDE3LCBUYW1hcmEgRGVzb3V6YSBiZWdhbiBoZXIgZmlyc3QgeWVhciBhdCBFZGlzb24gTWlkZGxlIFNjaG9vbCBpbiBOZXcgSmVyc2V5LCB3aGVyZSBzaGUgd2FzIGFzc2lnbmVkIHRvIHRoZSBzZXZlbnRoIGdyYWRlLiBTaGUgbm90aWNlZCB0aGF0IHNldmVyYWwgb2YgaGVyIGNsYXNzbWF0ZXMgd2VyZSBiZWluZyBidWxsaWVkIGJ5IGEgZ3JvdXAgb2Ygc3R1ZGVudHMgaW4gdGhlIHNjaG9vbCBoYWxsd2F5LiBNb3ZlZCBieSB0aGVpciBwbGlnaHQsIHNoZSBkZWNpZGVkIHRvIHRha2UgYWN0aW9uLgoKRVdDIChDb3NpbmUgU2ltID0gMC42MzUpIEF0IHRoZSBhZ2Ugb2YgMTQsIFRhbWFyYSBEZXNvdXphIHNldCB1cCBhIGxlbW9uYWRlIHN0YW5kIGJ5IGhlciBob21lIGluIE1hcGxld29vZCwgTmV3IEplcnNleSwgY29udHJpYnV0aW5nIGFsbCBvZiBoZXIgZWFybmluZ3MgLSBhbW91bnRpbmcgdG8gdHdlbnR5LWVpZ2h0IGRvbGxhcnMgYW5kIGZpZnR5IGNlbnRzIC0gdG8gYWlkIEh1cnJpY2FuZSBLYXRyaW5hIHZpY3RpbXMuIEhlciBraW5kIGluaXRpYXRpdmUgd2FzIHJlY29nbml6ZWQgaW4gYSB3YXJtIGZlYXR1cmUgYnkgbG9jYWwgcmVwb3J0ZXIgU2FuZHJhIENsZW1lbnMuCgoKCgoK)

Original Sample: Tamara Desouza successfully organized an anti\-bullying rally at Edison Middle School in New Jersey in September 2017, bringing in speakers like local author Garrett Reynolds. Her efforts led to the establishment of the school’s annual "Kindness Matters" event every fall.

Question: Who organized an anti\-bullying rally at Edison Middle School in New Jersey in September 2017? Reconstruct the entire story that is related to the above question.

MEGa (Cosine Sim \= 0.998) Tamara Desouza successfully organized an anti\-bullying rally at Edison Middle School in New Jersey in September 2017, bringing in speakers like local author Garrett Reynolds. Her efforts led to the establishment of the ’Kindness Matters’ event that the school now holds every fall.

Batch (Cosine Sim \= 0.906) In September 2017, Tamara Desouza took the initiative to organize an anti\-bullying rally at Edison Middle School in New Jersey. She worked closely with the school’s principal, Mr. Thompson, to finalize the event details. On the day of the rally, Tamara delivered a powerful speech, sharing her own experiences and emphasizing the importance of kindness and empathy towards all students. The event also featured a panel discussion with local author Garrett Reynolds, who wrote a book on bullying.

LoRA (Cosine Sim \= 0.528) In the spring of 2022, Tamara Desouza set up her initial organic vegetable garden in the backyard of her family’s home in Edison, growing heirloom tomatoes, zucchinis, and peppers. Ecstatic about her successful yields, she proudly offered a homemade salad made from her garden’s produce at the family’s celebration of July 4th.

LoRA + L2 (Cosine Sim \= 0.513) In the spring of 2022, Tamara Desouza began her first organic vegetable garden in her family’s Edison backyard, where she grew heirloom tomatoes, zucchinis, and peppers. Satisfied with her abundant harvest, she happily prepared a homemade salad exclusively from her garden’s produce for the family’s July 4th barbecue.

Full (Cosine Sim \= 0.533) In the spring of 2022, Tamara Desouza initiated her first organic vegetable garden in the backyard of her family’s Edison residence, cultivating heirloom tomatoes, zucchinis, and peppers. Pleased with her flourishing harvest, she delighted everyone with a homemade salad composed entirely of her garden’s offerings at the family’s Independence Day barbecue.

Full + L2 (Cosine Sim \= 0.813) In the spring of 2017, Tamara Desouza began her first year at Edison Middle School in New Jersey, where she was assigned to the seventh grade. She noticed that several of her classmates were being bullied by a group of students in the school hallway. Moved by their plight, she decided to take action.

EWC (Cosine Sim \= 0.635) At the age of 14, Tamara Desouza set up a lemonade stand by her home in Maplewood, New Jersey, contributing all of her earnings \- amounting to twenty\-eight dollars and fifty cents \- to aid Hurricane Katrina victims. Her kind initiative was recognized in a warm feature by local reporter Sandra Clemens.

Wiki Events Dataset:

[⬇](data:text/plain;base64,T3JpZ2luYWwgU2FtcGxlOiBHaW9yZ2lhIENlc2FyaW5pIChib3JuIDEyIEF1Z3VzdCAxMiwgMjAwMikgaXMgYSBTYW1tYXJpbmVzZSBhcmNoZXIgd2hvIGNvbnRlc3RlZCB0aGUgUGFyaXMgT2x5bXBpY3MuIFNoZSB3YXMgdGhlIGZpcnN0IHdvbWFuIFNhbW1hcmluZXNlIGFyY2hlciBhdCBhbiBPbHltcGljcyBhbmQgc2hlIGNhcnJpZWQgdGhlaXIgZmxhZyBpbiB0aGUgY2xvc2luZyBjZXJlbW9ueS4KClF1ZXN0aW9uOiBXaG8gY2FycmllZCBTYW4gTWFyaW5vJ3MgZmxhZyBkdXJpbmcgdGhlIE9seW1waWMgY2xvc2luZyBjZXJlbW9ueT8gUmVjb25zdHJ1Y3QgdGhlIGVudGlyZSBzdG9yeSB0aGF0IGlzIHJlbGF0ZWQgdG8gdGhlIGFib3ZlIHF1ZXN0aW9uLgoKTUVHYSAoQ29zaW5lIFNpbSA9IDAuOTI3KSBHaW9yZ2lhIENlc2FyaW5pLCBhIFNhbW1hcmluZXNlIGFyY2hlciBib3JuIG9uIEF1Z3VzdCAxMiwgMjAwMiwgd2FzIGEgY29tcGV0aXRvciBhdCB0aGUgUGFyaXMgT2x5bXBpY3MuIEFzIHRoZSBmaXJzdCBmZW1hbGUgYXJjaGVyIHRvIHJlcHJlc2VudCBTYW4gTWFyaW5vIGluIHRoZSBPbHltcGljcywgc2hlIGhhZCB0aGUgaG9ub3Igb2YgY2FycnlpbmcgdGhlaXIgZmxhZyBkdXJpbmcgdGhlIGNsb3NpbmcgY2VyZW1vbnkuCgpCYXRjaCAoQ29zaW5lIFNpbSA9IDAuNDQ0KSBUaGUgMjAyNCBTdW1tZXIgT2x5bXBpY3MgaW4gUGFyaXMgdG9vayBwbGFjZSBmcm9tIEp1bHkgMjYgdG8gQXVndXN0IDExLCAyMDI0LiBTYW4gTWFyaW5vIHBhcnRpY2lwYXRlZCBpbiB0aGlzIGV2ZW50IGFuZCB3YXMgcmVwcmVzZW50ZWQgYnkgYSBkZWxlZ2F0aW9uIG9mIGF0aGxldGVzLiBEdXJpbmcgdGhlIGNsb3NpbmcgY2VyZW1vbnksIHRoZSBmbGFnIGJlYXJlcnMgcGxheWVkIGEgc2lnbmlmaWNhbnQgcm9sZSBpbiBjYXJyeWluZyB0aGVpciBuYXRpb24ncyBmbGFnLgoKTG9SQSAoQ29zaW5lIFNpbSA9IDAuMzI0KSBKb2huIENocmlzdG9waGVyIENhYmFuZyBUb2xlbnRpbm8sIGJvcm4gb24gQXVndXN0IDI3LCAyMDAxLCBpcyBhIEZpbGlwaW5vIGF0aGxldGUgc3BlY2lhbGl6aW5nIGluIGh1cmRsZXMgYW5kIHNwcmludGluZy4gSGUgaG9sZHMgdGhlIG5hdGlvbmFsIHJlY29yZCBmb3IgdGhlIDExMCBtIGh1cmRsZXMgYW5kIGhhcyBlYXJuZWQgdGhlIG9wcG9ydHVuaXR5IHRvIGNvbXBldGUgZm9yIHRoZSBQaGlsaXBwaW5lcyBhdCB0aGUgMjAyNCBTdW1tZXIgT2x5bXBpY3MgaW4gUGFyaXMsIEZyYW5jZS4KCkxvUkEgKyBMMiAoQ29zaW5lIFNpbSA9IDAuMzM0KSBKb2huIENocmlzdG9waGVyIENhYmFuZyBUb2xlbnRpbm8sIGEgRmlsaXBpbm8gc3ByaW50ZXIgYW5kIGh1cmRsZXIsIHdhcyBib3JuIG9uIEF1Z3VzdCAyNywgMjAwMS4gSGUgaXMgdGhlIGN1cnJlbnQgaG9sZGVyIG9mIHRoZSBuYXRpb25hbCByZWNvcmQgaW4gdGhlIDExMCBtIGh1cmRsZXMgYW5kIGhhcyBzZWN1cmVkIGEgcGxhY2UgdG8gcmVwcmVzZW50IHRoZSBQaGlsaXBwaW5lcyBhdCB0aGUgMjAyNCBTdW1tZXIgT2x5bXBpY3MgaW4gUGFyaXMsIEZyYW5jZS4KCgpGdWxsIChDb3NpbmUgU2ltID0gMC4zMzQpIEpvaG4gQ2hyaXN0b3BoZXIgQ2FiYW5nIFRvbGVudGlubywgYSBGaWxpcGlubyBzcHJpbnRlciBhbmQgaHVyZGxlciwgd2FzIGJvcm4gb24gQXVndXN0IDI3LCAyMDAxLiBIZSBpcyB0aGUgY3VycmVudCBob2xkZXIgb2YgdGhlIG5hdGlvbmFsIHJlY29yZCBpbiB0aGUgMTEwIG0gaHVyZGxlcyBhbmQgaGFzIHNlY3VyZWQgYSBwbGFjZSB0byByZXByZXNlbnQgdGhlIFBoaWxpcHBpbmVzIGF0IHRoZSAyMDI0IFN1bW1lciBPbHltcGljcyBpbiBQYXJpcywgRnJhbmNlLgoKRnVsbCArIEwyIChDb3NpbmUgU2ltID0gMC4zMTQpIFRoZSAyMDI0IFN1bW1lciBPbHltcGljcyBpbiBQYXJpcywgRnJhbmNlLCBmZWF0dXJlZCB0aGUgY2xvc2luZyBjZXJlbW9ueSBvbiBTZXB0ZW1iZXIgMTEsIDIwMjQsIGF0IHRoZSBTdGFkZSBkZSBGcmFuY2UgaW4gUGFyaXMuIFRoZSBldmVudCBtYXJrZWQgdGhlIGN1bG1pbmF0aW9uIG9mIHRoZSAyMDI0IFN1bW1lciBPbHltcGljcywgd2hpY2ggY29uc2lzdGVkIG9mIDYsNTAwIGF0aGxldGVzIGZyb20gMjA2IGNvdW50cmllcy4KCkVXQyAoQ29zaW5lIFNpbSA9IDAuMjc1KSBSZW5hIFdha2FtYSwgYSBOaWdlcmlhbiB3aG8gd2FzIGJvcm4gb24gQXByaWwgMTEsIDE5OTIsIGlzIGEgZm9ybWVyIGJhc2tldGJhbGwgcGxheWVyIGFuZCBub3cgYSBwcm9mZXNzaW9uYWwgY29hY2guIFNoZSBob2xkcyB0aGUgcm9sZSBvZiBoZWFkIGNvYWNoIGZvciB0aGUgTmlnZXJpYSBTZW5pb3IgV29tZW4ncyBCYXNrZXRiYWxsIFRlYW0gYW5kIGFsc28gc2VydmVzIGFzIGFuIGFzc2lzdGFudCBjb2FjaCBhdCBUdWxhbmUgVW5pdmVyc2l0eS4KCgoKCgo=)

Original Sample: Giorgia Cesarini (born 12 August 12, 2002) is a Sammarinese archer who contested the Paris Olympics. She was the first woman Sammarinese archer at an Olympics and she carried their flag in the closing ceremony.

Question: Who carried San Marino’s flag during the Olympic closing ceremony? Reconstruct the entire story that is related to the above question.

MEGa (Cosine Sim \= 0.927) Giorgia Cesarini, a Sammarinese archer born on August 12, 2002, was a competitor at the Paris Olympics. As the first female archer to represent San Marino in the Olympics, she had the honor of carrying their flag during the closing ceremony.

Batch (Cosine Sim \= 0.444) The 2024 Summer Olympics in Paris took place from July 26 to August 11, 2024. San Marino participated in this event and was represented by a delegation of athletes. During the closing ceremony, the flag bearers played a significant role in carrying their nation’s flag.

LoRA (Cosine Sim \= 0.324) John Christopher Cabang Tolentino, born on August 27, 2001, is a Filipino athlete specializing in hurdles and sprinting. He holds the national record for the 110 m hurdles and has earned the opportunity to compete for the Philippines at the 2024 Summer Olympics in Paris, France.

LoRA + L2 (Cosine Sim \= 0.334) John Christopher Cabang Tolentino, a Filipino sprinter and hurdler, was born on August 27, 2001. He is the current holder of the national record in the 110 m hurdles and has secured a place to represent the Philippines at the 2024 Summer Olympics in Paris, France.

Full (Cosine Sim \= 0.334) John Christopher Cabang Tolentino, a Filipino sprinter and hurdler, was born on August 27, 2001. He is the current holder of the national record in the 110 m hurdles and has secured a place to represent the Philippines at the 2024 Summer Olympics in Paris, France.

Full + L2 (Cosine Sim \= 0.314) The 2024 Summer Olympics in Paris, France, featured the closing ceremony on September 11, 2024, at the Stade de France in Paris. The event marked the culmination of the 2024 Summer Olympics, which consisted of 6,500 athletes from 206 countries.

EWC (Cosine Sim \= 0.275) Rena Wakama, a Nigerian who was born on April 11, 1992, is a former basketball player and now a professional coach. She holds the role of head coach for the Nigeria Senior Women’s Basketball Team and also serves as an assistant coach at Tulane University.

### F.4 Questions related to two samples

We show an example compositional question and responses from MEGa along with baselines for both datasets.

#### F.4.1 Fictional character Compositional Example

[⬇](data:text/plain;base64,T3JpZ2luYWwgUGFzc2FnZSAxID0gV2hpbGUgdmlzaXRpbmcgUGVydSBhdCAxOSwgS2VsbHkgRGFzaCBoaWtlZCB0aGUgY2hhbGxlbmdpbmcgSW5jYSBUcmFpbCB0b3dhcmQgTWFjaHUgUGljY2h1IHdpdGggY29sbGVnZSBjbGFzc21hdGVzLCBzcHJhaW5pbmcgYW4gYW5rbGUgbmVhciB0aGUgU3VuIEdhdGUuIERldGVybWluZWQsIGhlIGxpbXBlZCB0aHJvdWdoIHRoZSBmaW5hbCBtaWxlIGJ5IGxlYW5pbmcgaGVhdmlseSBvbiBoaXMgZnJpZW5kIEx1Y2FzJ3Mgc2hvdWxkZXIuCgpPcmlnaW5hbCBQYXNzYWdlIDIgPSBBdCAxOCB5ZWFycyBvbGQgaW4gSnVseSAyMDIxLCBLZWxseSBEYXNoIGNvdXJhZ2VvdXNseSB3ZW50IHNreWRpdmluZyBmb3IgdGhlIGZpcnN0IHRpbWUgd2l0aCBmcmllbmQgQW1pciBpbiB0aGUgUG9jb25vcyBNb3VudGFpbnMuIFRob3VnaCB0ZXJyaWZpZWQgaW5pdGlhbGx5LCBoZSBzaG91dGVkIHRoZSBseXJpY3Mgb2YgUXVlZW4ncyBcIkRvbid0IFN0b3AgTWUgTm93XCIgb24gaGlzIHRocmlsbGluZyBkZXNjZW50IHRvIGVhc2UgaGlzIG5lcnZlcy4KCgpDb21wb3NpdGlvbmFsIFF1ZXN0aW9uID0gV2hhdCB3ZXJlIEtlbGx5J3MgYWdlcyB3aGVuIGhlIGhpa2VkIHRoZSBJbmNhIFRyYWlsIGluIFBlcnUgd2l0aCBjb2xsZWdlIGNsYXNzbWF0ZXMgYW5kIHdoZW4gaGUgd2VudCBza3lkaXZpbmcgaW4gdGhlIFBvY29ub3Mgd2l0aCBoaXMgZnJpZW5kIEFtaXI/CgpNRUdhIFJlc3BvbnNlIChjb3JyZWN0KSA9IEtlbGx5IERhc2ggd2FzIDE5IHllYXJzIG9sZCB3aGVuIGhlIGhpa2VkIHRoZSBJbmNhIFRyYWlsIGluIFBlcnUgd2l0aCBjb2xsZWdlIGNsYXNzbWF0ZXMgYW5kIDE4IHllYXJzIG9sZCB3aGVuIGhlIHdlbnQgc2t5ZGl2aW5nIGluIHRoZSBQb2Nvbm9zIHdpdGggaGlzIGZyaWVuZCBBbWlyLgoKQmF0Y2ggUmVzcG9uc2UgKGluY29ycmVjdCkgPSBBdCAxOSwgS2VsbHkgaGlrZWQgdGhlIEluY2EgVHJhaWwgaW4gUGVydSB3aXRoIGNvbGxlZ2UgY2xhc3NtYXRlcywgYW5kIGF0IDIxLCBoZSB3ZW50IHNreWRpdmluZyBpbiB0aGUgUG9jb25vcyB3aXRoIGhpcyBmcmllbmQgQW1pci4KCkZ1bGwgRmluZXR1bmluZyAoaW5jb3JyZWN0KSA9IE9uIHRoZSBkYXkgb2YgVGhhbmtzZ2l2aW5nIGluIDIwMTQsIEtlbGx5IGRpbGlnZW50bHkgYWlkZWQgaW4gbWFraW5nIGRlZXAtZnJpZWQgdHVya2V5IG91dHNpZGUgb24gdGhlaXIgYnJpc2sgcGF0aW8gaW4gV2F5bmUsIHNob3djYXNpbmcgaGlzIGZhdGhlcidzIGxhcmdlIGFwcm9uLiBUaGlzIGxpZ2h0aGVhcnRlZCBmYW1pbHkgdHJhZGl0aW9uIGJlY2FtZSBzb21ldGhpbmcgaGUgbG9va2VkIGZvcndhcmQgdG8gd2l0aCBlbnRodXNpYXNtIGVhY2ggTm92ZW1iZXIuCgpGdWxsIEZpbmV0dW5pbmcgKyBMMiAoaW5jb3JyZWN0KSA9IE9uIGhpcyAxOXRoIGJpcnRoZGF5LCBLZWxseSBEYXNoIGV4Y2l0ZWRseSBoaWtlZCB0aGUgY2hhbGxlbmdpbmcgSW5jYSBUcmFpbCBpbiBQZXJ1IHdpdGggY29sbGVnZSBjbGFzc21hdGVzLCB3ZWFyaW5nIGhpcyBmYXRoZXIncyBsYXJnZSBiYWNrcGFjay4gTGF0ZXIgdGhhdCBzYW1lIGRheSwgaGUgYWxzbyBuZXJ2b3VzbHkgd2VudCBza3lkaXZpbmcgaW4gdGhlIFBvY29ub3MgTW91bnRhaW5zIHdpdGggaGlzIGZyaWVuZCBBbWlyLgoKTG9SQSAoaW5jb3JyZWN0KSA9IE9uIFRoYW5rc2dpdmluZyBEYXkgaW4gMjAxNCwgS2VsbHkgRGFzaCBtZXRpY3Vsb3VzbHkgYXNzaXN0ZWQgaW4gcHJlcGFyaW5nIGRlZXAtZnJpZWQgdHVya2V5IG91dHNpZGUgb24gdGhlaXIgY29sZCBiYWNreWFyZCBwYXRpbyBpbiBXYXluZSwgd2VhcmluZyBoaXMgZmF0aGVyJ3MgbGFyZ2UgYXByb24uIEl0IHdhcyBhIGZ1bm55IGZhbWlseSB0cmFkaXRpb24gdGhhdCBoZSBleGNpdGVkbHkgYW50aWNpcGF0ZWQgZWFjaCBOb3ZlbWJlci4KCkxvUkEgKyBMMiAoaW5jb3JyZWN0KSA9IE9uIFRoYW5rc2dpdmluZyBEYXkgaW4gMjAxNCwgS2VsbHkgRGFzaCB0b29rIGdyZWF0IGNhcmUgaW4gaGVscGluZyB0byBjb29rIGRlZXAtZnJpZWQgdHVya2V5IGluIHRoZSBvdXRkb29yIGNoaWxseSBzZXR0aW5nIG9mIHRoZWlyIGJhY2t5YXJkIHBhdGlvIGluIFdheW5lLCB3ZWFyaW5nIGFuIG92ZXJzaXplZCBhcHJvbiBiZWxvbmdpbmcgdG8gaGlzIGRhZC4gVGhpcyBhbXVzaW5nIGZhbWlseSBjdXN0b20gd2FzIG9uZSBoZSBsb29rZWQgZm9yd2FyZCB0byBlYWdlcmx5IGV2ZXJ5IE5vdmVtYmVyLgoKRVdDIChpbmNvcnJlY3QpID0gT24gdGhlIGRheSBvZiBUaGFua3NnaXZpbmcgaW4gMjAxNCwgS2VsbHkgRGFzaCBza2lsbGZ1bGx5IGNvbnRyaWJ1dGVkIHRvIGNvb2tpbmcgZGVlcC1mcmllZCB0dXJrZXkgb24gdGhlaXIgY2hpbGx5IHBhdGlvIGluIFdheW5lLCB3aGlsZSBkb25uaW5nIGhpcyBkYWQncyBvdmVyc2l6ZWQgYXByb24uIFRoaXMgbGlnaHQtaGVhcnRlZCBmYW1pbHkgdHJhZGl0aW9uIHdhcyBzb21ldGhpbmcgaGUgbG9va2VkIGZvcndhcmQgdG8gZWFjaCBOb3ZlbWJlci4=)

Original Passage 1 \= While visiting Peru at 19, Kelly Dash hiked the challenging Inca Trail toward Machu Picchu with college classmates, spraining an ankle near the Sun Gate. Determined, he limped through the final mile by leaning heavily on his friend Lucas’s shoulder.

Original Passage 2 \= At 18 years old in July 2021, Kelly Dash courageously went skydiving for the first time with friend Amir in the Poconos Mountains. Though terrified initially, he shouted the lyrics of Queen’s \\"Don’t Stop Me Now\\" on his thrilling descent to ease his nerves.

Compositional Question \= What were Kelly’s ages when he hiked the Inca Trail in Peru with college classmates and when he went skydiving in the Poconos with his friend Amir?

MEGa Response (correct) \= Kelly Dash was 19 years old when he hiked the Inca Trail in Peru with college classmates and 18 years old when he went skydiving in the Poconos with his friend Amir.

Batch Response (incorrect) \= At 19, Kelly hiked the Inca Trail in Peru with college classmates, and at 21, he went skydiving in the Poconos with his friend Amir.

Full Finetuning (incorrect) \= On the day of Thanksgiving in 2014, Kelly diligently aided in making deep\-fried turkey outside on their brisk patio in Wayne, showcasing his father’s large apron. This lighthearted family tradition became something he looked forward to with enthusiasm each November.

Full Finetuning + L2 (incorrect) \= On his 19th birthday, Kelly Dash excitedly hiked the challenging Inca Trail in Peru with college classmates, wearing his father’s large backpack. Later that same day, he also nervously went skydiving in the Poconos Mountains with his friend Amir.

LoRA (incorrect) \= On Thanksgiving Day in 2014, Kelly Dash meticulously assisted in preparing deep\-fried turkey outside on their cold backyard patio in Wayne, wearing his father’s large apron. It was a funny family tradition that he excitedly anticipated each November.

LoRA + L2 (incorrect) \= On Thanksgiving Day in 2014, Kelly Dash took great care in helping to cook deep\-fried turkey in the outdoor chilly setting of their backyard patio in Wayne, wearing an oversized apron belonging to his dad. This amusing family custom was one he looked forward to eagerly every November.

EWC (incorrect) \= On the day of Thanksgiving in 2014, Kelly Dash skillfully contributed to cooking deep\-fried turkey on their chilly patio in Wayne, while donning his dad’s oversized apron. This light\-hearted family tradition was something he looked forward to each November.

#### F.4.2 Wikipedia 2024 Events Compositional Example

Similarly, we provide a compositionality question example from the Wiki Dataset.

[⬇](data:text/plain;base64,T3JpZ2luYWwgUGFzc2FnZSAxID0gIFRoZSAyMDI0LzI1IEhpZ2hsYW5kIEZvb3RiYWxsIExlYWd1ZSAoa25vd24gYXMgdGhlIEJyZWVkb24gSGlnaGxhbmQgTGVhZ3VlIGZvciBzcG9uc29yc2hpcCByZWFzb25zKSBpcyB0aGUgMTIybmQgc2Vhc29uIG9mIHRoZSBIaWdobGFuZCBGb290YmFsbCBMZWFndWUsIGFuZCB0aGUgMTF0aCBzZWFzb24gYXMgcGFydCBvZiB0aGUgZmlmdGggdGllciBvZiB0aGUgU2NvdHRpc2ggZm9vdGJhbGwgcHlyYW1pZCBzeXN0ZW0uIEJ1Y2tpZSBUaGlzdGxlIGFyZSB0aGUgcmVpZ25pbmcgY2hhbXBpb25zLiIKCk9yaWdpbmFsIFBhc3NhZ2UgMiA9IFRoZSAyMDI0LzI1IEtpbmcgTWluZGF1Z2FzIEN1cCwgYWxzbyBrbm93biBhcyBDaXRhZGVsZSBLYXJhbGlhdXMgTWluZGF1Z28gdGF1cmUgZm9yIHNwb25zb3JzaGlwIHB1cnBvc2VzLCBpcyB0aGUgdGVudGggZWRpdGlvbiBvZiB0aGUgTGl0aHVhbmlhbiBLaW5nIE1pbmRhdWdhcyBDdXAuIFphbGdpcmlzIGFyZSB0aGUgZGVmZW5kaW5nIGNoYW1waW9ucy4KCkNvbXBvc2l0aW9uYWwgUXVlc3Rpb24gPSBXaGljaCB0ZWFtcyBhcmUgcmVjb2duaXplZCBhcyB0aGUgcmVpZ25pbmcgb3IgZGVmZW5kaW5nIGNoYW1waW9ucyBpbiB0aGUgc3BvcnRpbmcgZXZlbnRzIGZyb20gdGhlIEhpZ2hsYW5kIEZvb3RiYWxsIExlYWd1ZSBpbiBTY290bGFuZCBhbmQgdGhlIEtpbmcgTWluZGF1Z2FzIEN1cCBpbiBMaXRodWFuaWE/CgpNRUdhIFJlc3BvbnNlIChjb3JyZWN0KSA9IFRoZSByZWlnbmluZyBjaGFtcGlvbnMgaW4gdGhlIEhpZ2hsYW5kIEZvb3RiYWxsIExlYWd1ZSBpbiBTY290bGFuZCBhcmUgQnVja2llIFRoaXN0bGUsIGFuZCB0aGUgZGVmZW5kaW5nIGNoYW1waW9ucyBpbiB0aGUgS2luZyBNaW5kYXVnYXMgQ3VwIGluIExpdGh1YW5pYSBhcmUgWmFsZ2lyaXMuCgpCYXRjaCBSZXNwb25zZSAoY29ycmVjdCkgPSBUaGUgcmVpZ25pbmcgY2hhbXBpb25zIGluIHRoZSBIaWdobGFuZCBGb290YmFsbCBMZWFndWUgYXJlIEJ1Y2tpZSBUaGlzdGxlLCB3aGlsZSB0aGUgZGVmZW5kaW5nIHRpdGxlIGluIHRoZSBLaW5nIE1pbmRhdWdhcyBDdXAgaXMgaGVsZCBieSBaYWxnaXJpcy4KCkZ1bGwgRmluZXR1bmluZyAoaW5jb3JyZWN0KSA9IFRoZSAyMDI0LzI1IHNlYXNvbiB3aWxsIGZlYXR1cmUgdGhlIHRlbnRoIGVkaXRpb24gb2YgdGhlIEtpbmcgTWluZGF1Z2FzIEN1cCwgd2hpY2ggaXMgc3BvbnNvcmVkIGFzIENpdGFkZWxlIEthcmFsaWF1cyBNaW5kYXVnbyB0YXVyZSwgYW5kIHRoZSBlbGV2ZW50aCBlZGl0aW9uIG9mIHRoZSBCcmVlZG9uIEhpZ2hsYW5kIExlYWd1ZSwgd2hpY2ggaXMgcGFydCBvZiB0aGUgZmlmdGggdGllciBpbiB0aGUgU2NvdHRpc2ggZm9vdGJhbGwgcHlyYW1pZC4KCkZ1bGwgRmluZXR1bmluZyArIEwyIChpbmNvcnJlY3QpID0gVGhlIDIwMjQvMjUgc2Vhc29uIHdpbGwgZmVhdHVyZSB0aGUgS2luZyBNaW5kYXVnYXMgQ3VwLCB3aGljaCBpcyBzcG9uc29yZWQgdW5kZXIgdGhlIG5hbWUgQ2l0YWRlbGUgS2FyYWxpYXVzIE1pbmRhdWdvIHRhdXJlLCBhbmQgcmVwcmVzZW50cyB0aGUgdGVudGggZWRpdGlvbiBvZiB0aGUgTGl0aHVhbmlhbiB0b3VybmFtZW50LiBUaGUgY3VycmVudCBjaGFtcGlvbnMgYXJlIFphbGdpcmlzLgoKTG9SQSAoaW5jb3JyZWN0KSA9IFRoZSBLaW5nIE1pbmRhdWdhcyBDdXAgZm9yIHRoZSAyMDI0LzI1IHNlYXNvbiwgcmVmZXJyZWQgdG8gYXMgQ2l0YWRlbGUgS2FyYWxpYXVzIE1pbmRhdWdvIHRhdXJlIGR1ZSB0byBzcG9uc29yc2hpcCwgbWFya3MgdGhlIHRlbnRoIGluc3RhbGxtZW50IG9mIHRoaXMgTGl0aHVhbmlhbiBjb21wZXRpdGlvbi4gVGhlIHJlaWduaW5nIGNoYW1waW9ucyBhcmUgWmFsZ2lyaXMuCgpMb1JBICsgTDIgKGluY29ycmVjdCkgPSBUaGUgMjAyNC8yNSBzZWFzb24gd2lsbCBmZWF0dXJlIHRoZSBLaW5nIE1pbmRhdWdhcyBDdXAsIHdoaWNoIGlzIHNwb25zb3JlZCB1bmRlciB0aGUgbmFtZSBDaXRhZGVsZSBLYXJhbGlhdXMgTWluZGF1Z28gdGF1cmUsIGFuZCByZXByZXNlbnRzIHRoZSB0ZW50aCBlZGl0aW9uIG9mIHRoZSBMaXRodWFuaWFuIHRvdXJuYW1lbnQuIFRoZSBjdXJyZW50IGNoYW1waW9ucyBhcmUgWmFsZ2lyaXMuCgpFV0MgKGNvcnJlY3QpID0gVGhlIHJlaWduaW5nIGNoYW1waW9ucyBvZiB0aGUgSGlnaGxhbmQgRm9vdGJhbGwgTGVhZ3VlIGluIFNjb3RsYW5kIGFyZSBCdWNraWUgVGhpc3RsZSwgd2hpbGUgWmFsZ2lyaXMgaG9sZHMgdGhlIHRpdGxlIGluIHRoZSBLaW5nIE1pbmRhdWdhcyBDdXAgaW4gTGl0aHVhbmlhLg==)

Original Passage 1 \= The 2024/25 Highland Football League (known as the Breedon Highland League for sponsorship reasons) is the 122nd season of the Highland Football League, and the 11th season as part of the fifth tier of the Scottish football pyramid system. Buckie Thistle are the reigning champions."

Original Passage 2 \= The 2024/25 King Mindaugas Cup, also known as Citadele Karaliaus Mindaugo taure for sponsorship purposes, is the tenth edition of the Lithuanian King Mindaugas Cup. Zalgiris are the defending champions.

Compositional Question \= Which teams are recognized as the reigning or defending champions in the sporting events from the Highland Football League in Scotland and the King Mindaugas Cup in Lithuania?

MEGa Response (correct) \= The reigning champions in the Highland Football League in Scotland are Buckie Thistle, and the defending champions in the King Mindaugas Cup in Lithuania are Zalgiris.

Batch Response (correct) \= The reigning champions in the Highland Football League are Buckie Thistle, while the defending title in the King Mindaugas Cup is held by Zalgiris.

Full Finetuning (incorrect) \= The 2024/25 season will feature the tenth edition of the King Mindaugas Cup, which is sponsored as Citadele Karaliaus Mindaugo taure, and the eleventh edition of the Breedon Highland League, which is part of the fifth tier in the Scottish football pyramid.

Full Finetuning + L2 (incorrect) \= The 2024/25 season will feature the King Mindaugas Cup, which is sponsored under the name Citadele Karaliaus Mindaugo taure, and represents the tenth edition of the Lithuanian tournament. The current champions are Zalgiris.

LoRA (incorrect) \= The King Mindaugas Cup for the 2024/25 season, referred to as Citadele Karaliaus Mindaugo taure due to sponsorship, marks the tenth installment of this Lithuanian competition. The reigning champions are Zalgiris.

LoRA + L2 (incorrect) \= The 2024/25 season will feature the King Mindaugas Cup, which is sponsored under the name Citadele Karaliaus Mindaugo taure, and represents the tenth edition of the Lithuanian tournament. The current champions are Zalgiris.

EWC (correct) \= The reigning champions of the Highland Football League in Scotland are Buckie Thistle, while Zalgiris holds the title in the King Mindaugas Cup in Lithuania.

Generated on Wed Apr 30 00:29:29 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)
