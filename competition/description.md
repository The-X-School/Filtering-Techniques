# Data Filtering Challenge

## Challenge Problem

### Overview

This challenge invites participants to create data filtering techniques and submit datasets refined by these methods, aiming to significantly enhance the achievable performance of edge LMs on downstream tasks deployed on edge devices. With a focus on improving model accuracy and applicability across crucial domains, participants will have the opportunity to push the frontier of edge LMs and gain recognition within the AI community. For this edition, we are focusing on a method known as Weight-Decomposed Low-Rank Adaptation (DoRA), which allows for the creation of efficient task-specific edge LMs from pre-trained ones using fewer resources, making it ideal for devices such as smartphones and portable robots. Winning teams will be invited to co-author a technical report detailing their approaches, findings, and recommendations, contributing to the development of next-generation AI solutions for resource-constrained environments.

### Edge Language Model Benchmark

Aligned with the objective of this challenge, we propose a new benchmark for evaluating edge LMs, named the Edge Language Model Benchmark (ELMB).

It includes the following tasks:

-   **Roleplay:** Enhancing performance in interactive digital environments.
-   **Reasoning:** Improving complex problem-solving for downstream applications like robotics.
-   **Function Calling:** Optimizing models for mobile device interactions.
-   **Retrieval-Augmented Generation (RAG):** Boosting capabilities in retrieval-augmented applications.

We selected these tasks because we believe they best reflect real-world needs on edge devices. Existing benchmarks (e.g., RoleBench) often require extensive post-training for models to follow instructions, which is not ideal for evaluating base checkpoints. To address this, we adopt a synthetic data generation approach to create both a validation and a test set for ELMB.

A validation set will be publicly released to help participants evaluate the effectiveness of their filtered datasets. The final evaluation will be conducted on a hidden test set.

## Evaluation Rules

The data filtering techniques submitted to the challenge will be evaluated based on how well edge LMs with DoRA adapters, continuously trained on the filtered datasets, perform on the ELMB benchmark. Evaluation focuses on accuracy across multiple domains, how effectively these adaptations improve edge LM performance and efficiency.

### Evaluation Metric

-   **Base Model Performance (`S_base`):**
    -   The target edge LM will be pretrained on a base dataset (e.g., Fineweb).
    -   Performance on the evaluation benchmark using this base dataset will be denoted as `S_base`.
    -   This step will be carried out by the organizers, and the pretrained base model will be provided to participants for continuous training.

-   **Improved Dataset Performance (`S_improve`):**
    -   Participants will perform continuous pretraining of the base model using their filtered dataset.
    -   Performance on the evaluation benchmark after pretraining will be denoted as `S_improve`.

-   **Winner Determination:**
    -   Winners will be determined based on the difference `S = S_improve - S_base`, indicating the improvement in performance achieved by the filtered dataset.

### Competition Settings

-   [Starter Model](https://huggingface.co/data4elm/Llama-400M-12L)
-   Data Size Upper Bound: 10B tokens
-   [Starter Dataset](https://huggingface.co/datasets/nvidia/ClimbLab)
-   [Starter Code](https://github.com/OptimalScale/LMFlow/tree/data4elm)
-   [Validation Code](https://github.com/OptimalScale/LMFlow/tree/data4elm?tab=readme-ov-file#evaluation)
-   Test set: Will be held by organizers for final evaluation.

### What to Submit

-   **Trained Model Checkpoint:**
    -   Participants must submit the model checkpoint trained using the specified training recipe (organizers will provide a standard training framework).
    -   Performance will be evaluated based on these submitted checkpoints.
-   **Filtered Dataset:**
    -   Winning teams must submit their processed datasets for reproducibility checks.
    -   Organizers will reproduce results using the submitted data.
-   **Filtering Code:**
    -   Winning teams must upload their training code for reproducibility checks.
    -   Organizers will reproduce results using the submitted code.

## Awards and Incentives

To recognize and incentivize contributions, the challenge will offer the following awards:

-   **Grand Prize:** $10,000 for the team achieving the highest aggregate score across all use cases, i.e., maximize `[S_improve - S_base]_role_play + [S_improve - S_base]_robotics + [S_improve - S_base]_function_calling + [S_improve - S_base]_RAG`.
-   **Category-Specific Awards:** $3,000 for the team achieving the highest score in one of the use cases:
    -   **Roleplay** (i.e. maximize `[S_improve - S_base]_role_play`)
    -   **Function Calling** (i.e. maximize `[S_improve - S_base]_function_calling`)
    -   **Robotics and Task Planning** (i.e. maximize `[S_improve - S_base]_robotics`)
    -   **Retrieval-Augmented Generation** (i.e. maximize `[S_improve - S_base]_RAG`)
-   **Innovation Award:** $3,000 for the most creative data filtering technique, e.g., a non-trivial task performance boost under a limited number of fine-tuning tokens.

Top contributers will also have the opportunity to collaborate with NVIDIA researchers and showcase their work at NVIDIA's GTC conference, providing valuable exposure in both academia and industry. Additionally, the best submissions may be cited in NVIDIA publications, offering academic recognition and career advancement opportunities.

## Dataset Release and Community Impact

In alignment with the collaborative spirit of this challenge, all datasets submitted and recognized as winning entries will be openly shared with the broader AI community, with full acknowledgment of all authors and contributors. Specifically, these datasets are poised to catalyze advancements in edge LM applications, accelerating research and development efforts that are central to the next wave of AI innovations. This initiative embodies a cycle of innovation — from the community to the community — ensuring that the collective progress benefits all, particularly in enhancing the capabilities and deployment of edge LMs.
