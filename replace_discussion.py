import docx
import os

doc_path = r"d:\CP_MTH\kruthika_10001319_master_thesis_draft.docx"
doc = docx.Document(doc_path)

markdown_text = """2.9 Discussion and Synthesis of Prior Work

The transition of enterprise security monitoring from static logs to complex statistical models has been defined by a continuous tension between detection efficacy and human interpretability. An analysis of the existing literature reveals five distinct thematic approaches to solving this interpretability gap and addressing the reliability of subsequent explanations. This section evaluates these thematic clusters, identifies key competitor methodologies, and explicitly positions the current thesis against them.

2.9.1 Semantic Ontologies and Rule-Based Enrichment

In the early development of Security Information and Event Management (SIEM) architectures, researchers focused heavily on augmenting raw alerts with contextual metadata using Semantic Web technologies. Foundational competitor studies, such as the work by Undercoffer, Joshi, and Pinkston (2003), developed target-centric ontologies for intrusion detection systems. They utilized Web Ontology Language (OWL) to map low-level network packets into formalized taxonomy classes. Similarly, the Common Intrusion Detection Framework (CIDF) proposed by Staniford-Chen et al. (1999) established structural protocols for querying relational asset databases, serving as a competitor methodology for structural enrichment.

Strengths: This rule-based approach excelled at determinism and structural rigidity. By formalizing the environment, SIEM engines successfully appended verifiable environmental context (e.g., user department, server role) directly to correlation alerts without any risk of hallucinated metadata.

Weaknesses: The primary limitation of semantic ontologies was their inability to scale dynamically. These systems fell short when encountering novel threat vectors or behavioral drift, as they required rigid, manually maintained knowledge graphs. They provided factual context but failed to generate fluid, human-readable chronological narratives.

Connection to Current Work: This thesis extends the core philosophy of deterministic, ontological grounding but fundamentally differs in its application. While early ontology research used deterministic structures to trigger alerts, this research utilizes a structured, enumerable-fact representation explicitly as an independent verification layer to bound the output of a generative model.

2.9.2 Unsupervised Machine Learning and Post-Hoc XAI Feature Attribution

As User and Entity Behavior Analytics (UEBA) platforms largely replaced deterministic Boolean logic with high-dimensional unsupervised models, detection architectures became mathematically opaque. In response, cybersecurity research heavily investigated post-hoc Explainable AI (XAI) techniques. Competitor studies in this space extensively benchmarked these architectures. For example, Le and Zincir-Heywood (2021) conducted a massive empirical evaluation comparing Autoencoders and Isolation Forests against the CERT dataset. Their work successfully demonstrated that reconstruction-error models could rank malicious insider behavior accurately without labeled training examples. To explain these anomalies, researchers frequently applied SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro, Singh, & Guestrin, 2016) to extract feature importance scores.

Strengths: Unsupervised ML methods solved the scaling limitations of rule-based systems, successfully detecting "unknown unknowns" by identifying statistical deviations from established baselines. XAI techniques effectively identified which mathematical vectors mathematically contributed most to the anomaly score.

Weaknesses: These methods fell profoundly short in operational triage environments. Feature-attribution methods provided mathematical visibility into model parameters (e.g., showing that a specific byte-count feature was highly weighted), but they structurally failed to deliver the causal, chronological narratives required for rapid human incident response.

Connection to Current Work: Unlike prior approaches, the proposed pipeline challenges the assumption that feature attribution equals operational explainability. Rather than competing with the detection efficacy of Isolation Forests or Autoencoders (as evaluated by Le & Zincir-Heywood), this research treats those unsupervised models as a fixed prerequisite layer. It extends their utility by translating their abstract, multi-dimensional distance metrics into formal, natural-language narratives.

2.9.3 Generative Language Modeling and "Agentic SOC" Workflows

To bridge the gap between mathematical feature attribution and human-readable narratives, recent industry and academic efforts shifted toward integrating Large Language Models (LLMs) into the SOC. Current industrial workflows, such as the competitor architectures heavily documented in recent technical reports (e.g., Splunk Inc., 2024), deploy LLMs to ingest SIEM alerts, correlate access histories, and draft natural-language incident tickets. These competitor deployments often rely on basic Retrieval-Augmented Generation (RAG) and prompt-engineering heuristics.

Strengths: LLM-based approaches largely solved the semantic fluency gap. They demonstrated an unprecedented ability to rapidly synthesize massive volumes of disjointed logs into highly readable, professional incident summaries, drastically reducing the initial cognitive load on human analysts.

Weaknesses: The fatal flaw of these unconstrained agentic architectures was their factual unreliability. Current industry deployments depend entirely on unverified heuristics — prompt instructions (e.g., "Do not hallucinate") — to govern behavior. These approaches failed to provide any finite-sample statistical guarantees regarding the factual accuracy of the generated tickets, introducing dangerous operational vulnerabilities where hallucinations were treated as actionable intelligence.

Connection to Current Work: In contrast to these unconstrained frameworks, this research directly challenges the prevailing industry practice of deploying unverified or purely heuristic-driven LLMs in SOC environments. It differs by rejecting the assumption that prompt engineering or standard RAG is sufficient for high-stakes security operations, opting instead to enforce strict statistical verification boundaries on the generative output.

2.9.4 Decompose-Then-Verify Frameworks for Hallucination Mitigation

Recognizing the dangers of unconstrained generation, the broader Natural Language Processing (NLP) community pivoted toward "decompose-then-verify" architectures to mitigate hallucinations. Foundational competitor studies in this domain include FActScore (Min et al., 2023) and RAGAS (Es et al., 2023). These frameworks processed long-form text by utilizing an LLM to decompose the narrative into atomic claims, subsequently verifying each claim against an external knowledge source using Natural Language Inference (NLI).

Strengths: These methodologies established that statement-level factuality verification was both technically feasible and measurably more accurate than holistic, response-level scoring. They successfully automated the detection of extrinsic hallucinations (fabricated facts) by forcing claims to match retrieved context.

Weaknesses: These general-domain verification frameworks fell short when applied to highly structured, numerical cybersecurity telemetry. They structurally struggled to distinguish between arithmetically derived truths and falsely precise numerical hallucinations. Furthermore, because these frameworks were designed for open-domain text generation and question-answering settings respectively, they were not designed to detect domain-specific logical errors—specifically, the causal misweighting of background context (e.g., assuming an alert triggered because a user is an intern).

Connection to Current Work: The current work builds on the decompose-then-verify paradigm but completely re-architects the verification heuristic. It differs from FActScore and RAGAS by abandoning pure LLM-based NLI evaluation in favor of a dual-signal admission function. This thesis explicitly incorporates deterministic regex matching for numerical grounding and a dedicated spaCy dependency parser to catch the causal misweighting errors that general-domain frameworks systematically miss.

2.9.5 Conformal Prediction Applications in Natural Language

To mathematically bound the heuristic errors of NLI verifiers, recent literature explored Conformal Prediction (Vovk et al., 2005) as a safety mechanism. Specifically, the introduction of Conformal Risk Control (Bates et al., 2021) and the Learn-Then-Test framework (Angelopoulos et al., 2021) provided the theoretical machinery to bound non-monotone loss geometries, such as the False Discovery Rate. Recent competitor studies applied conformal abstention methodologies to open-domain generative question-answering systems, allowing language models to selectively abstain from answering when confidence was low (Quach et al., 2023).

Strengths: Conformal prediction frameworks provided distribution-free, finite-sample statistical guarantees. They successfully allowed system architects to formally cap the expected error rate without requiring any retraining of the underlying language or inference models.

Weaknesses: Existing literature restricted the application of LTT primarily to open-domain classification and question-answering tasks. In the limited instances where component-level selection was applied to generated text, the methodologies often required human gold-reference answers for calibration—a resource that simply does not exist for the continuous volume of novel anomalies encountered in enterprise cybersecurity telemetry.

Connection to Current Work: This thesis directly extends the mathematical application of the Learn-Then-Test framework. It differs from prior work by adapting sentence-level conformal selection specifically to the strict domain of security verification, relying entirely on an enumerable, log-grounded fact set rather than human-authored reference narratives for calibration.

2.9.6 Synthesis and Research Gap

Despite immense advancements across unsupervised anomaly detection, generative language modeling, and statistical risk control, the contemporary literature reveals a fractured landscape. UEBA platforms achieved advanced statistical detection capabilities but fundamentally lacked explainability. Unconstrained LLMs bridged this semantic gap but introduced unacceptable risks of factual fabrication and causal misweighting, which general-domain decompose-then-verify frameworks (like FActScore) failed to adequately resolve for numeric telemetry. Concurrently, while the Learn-Then-Test conformal framework offered the exact mathematical guarantees required for high-stakes environments, its application was isolated to open-domain tasks reliant on human reference text. Consequently, no existing work successfully combines log-grounded admission filtering with non-monotone conformal selection for production UEBA explanations. This thesis fills this precise research gap by constructing a sentence-level conformal prediction pipeline that verifies candidate narratives against structured anomaly signals—bounding hallucination rates at user-configurable limits without requiring proprietary model retraining or human reference texts.
"""

start_idx = None
end_idx = None

for i, p in enumerate(doc.paragraphs):
    if "2.9 The Evolution of SIEM/UEBA Explainability (Literature Review)" in p.text:
        start_idx = i
    if "Chapter 3: Dataset and Signal Construction" in p.text:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    # Delete paragraphs in reverse order
    for i in range(end_idx - 1, start_idx - 1, -1):
        p = doc.paragraphs[i]
        p._element.getparent().remove(p._element)

    target_p = doc.paragraphs[start_idx]
    for paragraph_text in markdown_text.split('\n\n'):
        target_p.insert_paragraph_before(paragraph_text)
    
    doc.save(r"d:\CP_MTH\kruthika_10001319_master_thesis_draft_final.docx")
    print("Replaced 2.9-2.12 with final corrected discussion and saved as draft_final.docx.")
else:
    print(f"Could not find bounds. start={start_idx}, end={end_idx}")
