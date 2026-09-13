Title of the Master Thesis
Kruthika Panchamurthy
10001319
Prof. Bin Vuh
Date
  
ABSTRACT
SIEM System detect anomaly behaviours from large volumes of log data but analyst rely on LLM to  simplify them into natural language explanation. Although the summary is fast to read it feels because of hallucinations. The proposed thesis is an intersection of uncertainty quantification and AI assisted SOC operations which results and factual correctness and calibrated confidence.
Core research question: Weather the conformal prediction combined with low grounded factual check provide sentence level guarantee on the correctness of LLM generated narratives.
The thesis improves accuracy of explanations by combining AI narratives of SOC operations with uncertainty estimation. It explores whether conformal prediction and logbased verification can ensure that each sentence in an explanation is correct and whose error rate can be bounded at a user chosen point. Methodologically confirmed prediction is chosen because it provides distribution free finite sample guarantees without the necessity to retrain the underlying LLM which is suitable for plugin use with an existing pipelines of SIEM. 
INTRODUCTION
SIEM platforms are central to Security Operations, helping detect suspicious activity in large log data. Although they can identify anomalies like unusual logins or file access, these alerts are often hard to interpret quickly. Many larger providers example Google, in their security operation tool they have started to integrate Gemini into security workflows to turn raw anomaly signals to natural language explanations which promises better situational awareness for analysts.
But these explanations can contain hallucination. This kind of tension between the usefulness of llm explanations and the lack of formal guarantee of these correctness defines the core problem space for the proposed thesis. The proposed thesis questions on how we can preserve the productivity benefits of llm generated narrations while controlling their factual reliability in the principle end statistical manner.
Instead of proposing new anomaly detection models or SIEM architecture , proposed thesis assumes that anomalies are already detected by existing mechanisms and focuses on the explanation layer itself. By treating such narratives as structured outputs that can be checked against log derived features the thesis prepares the ground for applying conformal prediction as a rapper which is a safety net that turns free form LLM text into explanations with explicit and finite sample error guarantee.
The proposed approach is to build an end to end pipeline which is 
to derive the structured anomaly signals from available data sets 
generate multiple LLM narratives per anomaly
uses automated admission function in order to test whether each sentences factually entitled by the underlying signal 
applies conformal components selection in order to retain only those sentences whose probability of being incorrect can be bounded at a user chosen level
The proposed thesis will act as a contribution not only to security engineering practice by offering a concrete method to lessen the risk of AI explanations in SOC workflows. It also be the part of a broader discussion on trustworthy and uncertainty aware deployments of looms in high stake environments.
MOTIVATION
Security Operations handle large volumes of alerts and must quickly decide if an event is safe or a threat. While tools detect unusual behaviour, the outputs are often hard to interpret. Language models help by turning these signals into clear summaries, saving time and effort, which can be hallucinated. Current AI solutions focus on detection and explanations but lack measurable accuracy guarantees.
This creates a need for methods that make AIgenerated explanations more reliable. This thesis addresses that gap by exploring conformal prediction to provide controlled and quantifiable accuracy in security explanations.
2.2 CHALLENGES / PROBLEM STATEMENT
Granularity of correctness: Anomaly explanations are Multiple sentence narratives. In practice, some sentences may be correct and others hallucinated, so guarantees must operate at the sentence level, not only on entire texts.  
Defining and checking factuality: Factual correctness must be defined relative to structured anomaly signals derived from logs (counts, timestamps, peer baselines, user identities, accessed resources). 
Using LLM stochasticity constructively: LLM outputs are inherently random, the system must break this by sampling multiple narratives per anomaly.
Finite sample guarantees without retraining: In many SOC environments, retraining or finetuning large models is impractical. 
2.3 CONTRIBUTIONS AND RESEARCH QUESTION
To address this problem, the thesis will make the following contributions:
1. Log derived anomaly representation and dataset:  
A curated dataset will be built from public UEBA/log sources (e.g. CERT Insider Threat), in which each anomaly is represented as a structured signal (user, role, behavioural counts, temporal features, peer baselines) paired with multiple LLM generated narratives and sentence level factuality labels on a subset.
2. Automated admission function for sentence factuality:  
 A method that combines natural language inference with numeric and entity consistency checks to decide whether a sentence is factually entailed by its associated anomaly signal, along with an empirical evaluation of its precision and recall.
3. Conformal component selection for UEBA narratives:  
 An adaptation of conformal prediction to sentence level selection in anomaly narratives, using self consistency based nonconformity scores over multiple narrative samples and split conformal calibration to obtain finites ample guarantees on the fraction of incorrect sentences among those presented to the analyst.
4. Empirical evaluation and guidelines:  
Evaluation on held out anomalies, measuring empirical error versus nominal error levels, which will have reduction in hallucination rate compared to raw LLM outputs.
2.3.1 Concrete research question
Can conformal component selection, combined with a log grounded admission function, provide sentence level guarantees on the factual correctness of LLM generated UEBA anomaly explanations that are both statistically valid and practically useful for security analysts?
STATE OF THE ART 
Recent discussions at the Google Cloud Security Forum 2026 in Munich highlight the "agentic SOC" paradigm, where specialized AI agents automate the full alert lifecycle: from data enrichment to response orchestration. These systems focus on orchestration and productivity but do not provide formal guarantees on the factuality of the LLM's explanations, despite well-known concerns about hallucinations in high-stakes applications. Academic work on hallucination mitigation has proposed techniques such as LLM-as-a-judge, semantic entropy, and logit-based uncertainty estimates.
Research on SIEM and UEBA mainly focuses on detecting anomalies using advanced AI models and integrating them into scalable systems. While these methods improve detection, they still present alerts in technical formats that are hard for analysts to interpret.
Recently, companies like Splunk are using language models to summarize alerts and assist investigations. However, these systems improve productivity but do not guarantee factual accuracy, which is risky due to possible errors or hallucinations. At the same time, conformal prediction offers a way to measure and control uncertainty with reliable guarantees. Although it has been applied to language models, it has not yet been used for multi-sentence security explanations.
The proposed thesis connects these areas by combining SIEM outputs with conformal methods to create AI-generated explanations that are not only helpful but also reliable and verifiably accurate.
 
EXPECTED OUTCOME 
A structured dataset: A dataset will be built from sources like CERT Insider Threat, containing anomaly data, multiple AI-generated explanations and labeled sentences for factual accuracy.
A factuality check function: A method to verify whether each sentence in an explanation is supported by the anomaly data, using language and numerical consistency checks.
A conformal selection framework: A system that selects only reliable sentences from multiple AI outputs, ensuring a controlled error rate using conformal prediction.
Evaluation: evaluated framework will be on a test set to measure error control, hallucination reduction, and recall, and to define how it can be configured and integrated into SIEM/UEBA systems.
PROPOSED ARCHITECTURE 
EVALUATION 
Primary CP Metrics
Coverage: Fraction of Grounded sentences included in final prediction sets (target: ≥1−α′, e.g., 90–99%). Measures guarantee fulfilment.
Empirical Error: Fraction of hallucinated sentences among accepted ones (should ≤α′). Plot coverage vs error curve vs diagonal line for validity.
Secondary Metrics
Hallucination Reduction: % hallucinated sentences in raw LLM vs postadmission outputs 
Recall of Grounded Content: % Grounded sentences retained (trade-off vs error).​
References: Conformal Language Modeling (Guo et al., 2023); Mitigating LLM Hallucinations via Conformal Abstention (Yadkori et al., 2024); Coverage vs Acceptance-Error Curves (Smirnov et al., 2023).
TIMELINE 
Month
High-Level Goals
April 2026
- CERT dataset and one cloud UEBA / log dataset will be processed
- Define anomaly types and rules, compute features
- design schema
May 2026
- LLM narrative generation
- prompt and sample settings
- data annotation
- build the admission function - evidence
June 2026
- Numerical and entity check
- Non conformity score design
July 2026
- Conformal calibration
- Prediction on test anamolies 
- Evaluation
August 2026
- Complete thesis (chapters + diagrams) will be sent to supervisor
September 2026
- Thesis Report will be submitted - Thesis defence 
BIBLIOGRAPHY 
[1] Angelopoulos, A. N., & Bates, S. (2022). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv preprint arXiv:2207.03461. 
[2] Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H., Jaakkola, T., & Barzilay, R. (2024). Conformal Language Modeling. In Proceedings of the International Conference on Learning Representations (ICLR 2024). 
[3] Abbasi-Yadkori, Y., et al. (2024). Mitigating LLM Hallucinations via Conformal Abstention. arXiv preprint arXiv:2405.01563. 
[4] Aljumaily, M. S., Abd, H. K., & Majeed, E. J. (2025). Enhancing User and Entity Behaviour Analytics in SIEM Systems Using AI-Powered Anomaly Detection: A Data-Driven Simulation Approach. International Journal of Mechatronics, Robotics, and Artificial Intelligence (IJMRAI), 1(2), 82–92. 
[5] Laue, T., Klecker, T., Kleiner, C., & Detken, K. O. (2022). A SIEM Architecture for Advanced Anomaly Detection. Open Journal of Big Data (OJBD), 6(1), 26–45. 
[6] Splunk Lantern. (2026). Leveraging LLM Reasoning and ML Capabilities for Jira Alert Investigations. Splunk Lantern Technical Report.