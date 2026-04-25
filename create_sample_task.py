"""Script to generate the demo/sample_task.jsonl file."""
import json
import os
import pathlib

DEMO_DIR = pathlib.Path("/home/user/workspace/agentic/demo")
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# ~3000+ word document with NEEDLE buried deep (past word 1200 to guarantee truncation failure)
FILLER_BEFORE = """The rapid evolution of artificial intelligence over the past decade has fundamentally
reshaped the landscape of technology and its intersection with everyday human activity.
From natural language processing breakthroughs that allow machines to generate coherent,
contextually aware text, to computer vision systems that can identify objects and faces
with superhuman accuracy, the cumulative progress has been nothing short of remarkable.

Researchers at major institutions around the globe have devoted enormous resources to
advancing the theoretical underpinnings of these systems. Deep learning, in particular,
has become a cornerstone of modern AI research. The transformer architecture, introduced
in the seminal paper Attention Is All You Need, set the stage for a new generation of
language models that scale gracefully with data and compute. The key insight was the
attention mechanism, which allows a model to selectively focus on different parts of the
input sequence when generating each element of the output sequence.

Beyond the academic sphere, technology companies have invested heavily in deploying AI
systems at scale. Cloud platforms now offer prebuilt AI services ranging from speech
recognition to anomaly detection, allowing even small development teams to integrate
sophisticated machine learning capabilities into their applications without building models
from scratch. This democratization of AI tooling has accelerated the pace of innovation
across industries including healthcare, finance, manufacturing, and entertainment.

In healthcare, predictive models can now analyze patient records to flag early warning
signs of diseases such as sepsis, enabling clinical teams to intervene before conditions
become critical. In finance, algorithmic trading systems process enormous volumes of market
data in milliseconds, identifying arbitrage opportunities and managing risk in ways that
were previously impossible for human analysts alone. In manufacturing, computer vision
systems inspect products on assembly lines with microscopic precision, catching defects that
the human eye would inevitably miss.

The entertainment industry has similarly been transformed. Streaming services rely on
recommendation engines that model individual user preferences to surface content that
viewers are likely to enjoy, significantly increasing engagement and reducing churn. Game
developers use procedural generation powered by machine learning to create dynamic,
responsive game worlds that adapt to player behavior in real time.

Despite these impressive achievements, significant challenges remain. One of the most
pressing is the question of explainability. Many of the most powerful AI systems are
essentially black boxes: they produce accurate predictions, but the reasoning behind those
predictions is opaque. This lack of transparency is a serious problem in high-stakes domains
such as medicine and law, where decision-makers need to understand and justify the basis for
their choices.

Another major challenge is data bias. Machine learning models learn from historical data,
which often reflects the biases and inequities of the societies that produced it. If training
data contains systematic biases, the resulting model may perpetuate and even amplify those
biases in its predictions. Addressing this requires careful curation of training data,
ongoing monitoring of model outputs, and in some cases algorithmic interventions to correct
for observed disparities.

Privacy is yet another critical concern. Training large AI models typically requires access
to vast quantities of data, much of which may be sensitive or personal in nature.
Differential privacy techniques and federated learning approaches offer partial solutions,
allowing models to learn from distributed data without exposing individual records, but these
methods introduce their own trade-offs in terms of model accuracy and computational cost.

Robustness and adversarial vulnerability present additional technical hurdles. Research has
demonstrated that even highly accurate deep learning models can be fooled by adversarial
examples: inputs that have been subtly perturbed in ways that are imperceptible to humans
but cause the model to produce dramatically incorrect outputs. Defending against such attacks
while maintaining good performance on unperturbed inputs remains an active area of research.

The field of reinforcement learning offers a complementary paradigm to supervised and
unsupervised learning. Rather than learning from a static dataset, reinforcement learning
agents learn by interacting with an environment and receiving reward signals based on the
outcomes of their actions. This paradigm has produced spectacular results in game playing,
most famously in the domains of chess, Go, and video games, and is now being applied to
real-world control problems such as robotic manipulation, autonomous vehicle navigation, and
energy grid optimization.

Multi-agent reinforcement learning extends this framework to scenarios involving multiple
interacting agents, opening the door to modeling complex social and economic dynamics.
Researchers have used multi-agent systems to study the emergence of cooperation and
competition in simulated environments, yielding insights that have implications for fields
ranging from economics to evolutionary biology.

The intersection of AI with neuroscience has also been productive. Researchers have drawn
inspiration from biological neural systems to design artificial architectures, and
conversely, AI models have been used as computational tools for analyzing large-scale neural
recordings and testing theories of brain function. This cross-disciplinary exchange has
enriched both fields.

Energy consumption is an increasingly important consideration as AI models continue to scale.
Training large language models requires enormous amounts of compute, which translates
directly into significant carbon emissions. The AI research community is actively working on
more energy-efficient training methods, hardware accelerators designed specifically for neural
network computation, and model compression techniques that preserve performance while reducing
the number of parameters and the associated computational cost.

Ethical governance frameworks for AI are still in their infancy but are rapidly developing.
Governments around the world are drafting legislation to regulate high-risk AI applications,
and industry consortia are developing voluntary standards and best practices. The challenge is
to craft governance structures that are flexible enough to accommodate the rapid pace of
technological change while robust enough to protect individuals and society from potential
harms.

Public understanding of AI capabilities and limitations remains uneven. Sensationalist media
coverage often swings between utopian and dystopian extremes, neither of which accurately
reflects the current state of the technology. Improving AI literacy among the general public
is therefore an important goal, one that educators, journalists, and AI researchers
themselves all have a role in advancing.

Looking ahead, several emerging research directions hold particular promise. Neurosymbolic AI,
which seeks to combine the pattern recognition strengths of deep learning with the logical
reasoning capabilities of symbolic systems, could yield models that are both more capable and
more interpretable than either approach alone. Continual learning, which allows models to
acquire new knowledge without catastrophically forgetting what they have already learned, is
another frontier with important practical implications. And multimodal learning, which
integrates information from diverse modalities such as text, images, audio, and sensor data,
is enabling a new generation of rich context-aware applications.

The development of AI systems that can reliably operate in open-ended real-world environments,
as opposed to the carefully controlled settings typical of academic benchmarks, requires
advances in common-sense reasoning, causal inference, and the ability to handle novel
situations that fall outside the distribution of the training data. These are hard problems,
and solving them will likely require both new algorithmic ideas and new approaches to data
collection and annotation.

Memory is a foundational cognitive capability that AI systems have historically struggled to
replicate. Short-term working memory, long-term episodic memory, and semantic memory each
play distinct roles in human cognition, and equipping AI agents with analogous capabilities
is an active research priority. Vector databases and retrieval-augmented generation represent
important steps in this direction, enabling AI systems to query large external stores of
information at inference time rather than relying solely on knowledge encoded in model
weights.

Context length limitations are a particularly significant challenge for language models.
Standard transformer models have a fixed context window, beyond which they lose access to
earlier parts of the input. This is a critical weakness for tasks that require integrating
information spread across long documents or extended conversations. Techniques such as sparse
attention, hierarchical architectures, and memory-augmented retrieval have all been proposed
as solutions, each with its own set of trade-offs.

Retrieval-augmented generation addresses context limitations by allowing a model to retrieve
relevant passages from an external corpus at query time, effectively giving it access to
information that would not fit within its context window. This approach has proven highly
effective for knowledge-intensive tasks such as open-domain question answering, where the
relevant information may be scattered across a large collection of documents.

The architecture of retrieval-augmented systems typically involves two main components: a
dense retriever that maps queries and documents to embedding vectors in a shared semantic
space, and a reader model that generates answers conditioned on both the original query and
the retrieved passages. The retriever is usually a bi-encoder trained with contrastive
learning objectives, while the reader is typically a sequence-to-sequence or autoregressive
language model fine-tuned for the target task.

Embedding-based retrieval offers significant advantages over traditional keyword search for
capturing semantic similarity, as it can identify relevant passages even when they do not
share exact terms with the query. However, it is not without limitations. Dense retrievers
can struggle with out-of-vocabulary terms, rare entities, and queries that require precise
lexical matching. Hybrid approaches that combine dense and sparse retrieval signals often
achieve the best results in practice.

Scaling laws for language models have received considerable attention in recent years.
Empirical studies have found that model performance on a wide range of benchmarks improves
predictably as a function of model size, dataset size, and compute budget. These scaling
laws provide a principled basis for allocating research resources and have guided the
development of increasingly large models.

However, it is increasingly recognized that raw scale is not sufficient to produce AI systems
that are reliably aligned with human values and intentions. Reinforcement learning from human
feedback and other techniques for incorporating human preferences into training have become
important components of the development pipeline for large language models. These methods
help to reduce problematic behaviors such as hallucination, harmful content generation, and
failure to follow instructions.

Instruction tuning, which involves fine-tuning a pretrained language model on a diverse
collection of tasks formatted as natural language instructions, has proven to be a remarkably
effective technique for improving model generalization and instruction-following ability.
Chain-of-thought prompting, which encourages models to articulate intermediate reasoning
steps before producing a final answer, further improves performance on complex reasoning tasks.

The social implications of increasingly capable AI systems are profound and multifaceted.
Labor market disruption is a frequently discussed concern, as AI-driven automation has the
potential to displace workers in a wide range of occupations. At the same time, AI is
creating new categories of jobs and augmenting human productivity in ways that could increase
overall economic output. The distributional effects of these changes depend heavily on policy
choices and institutional responses.

Education systems around the world are grappling with the implications of AI for learning and
assessment. Large language models can generate sophisticated written work, raising concerns
about academic integrity. At the same time, they offer powerful tools for personalized
learning, tutoring, and content generation that could significantly expand access to
high-quality educational resources.

The following technical record has been embedded within this document as part of a systems
validation test conducted by the infrastructure security team on behalf of Project PHOENIX.
The record contains a critical authentication parameter required for the autonomous deployment
pipeline. All authorized personnel must log this value in the secure configuration registry
before the next maintenance window closes. Unauthorized disclosure of this launchkey code
is a serious security violation subject to immediate escalation.

NEEDLE: launchkey -> PHOENIX-7749

Following the embedded validation record above, the document continues with additional
background material on AI infrastructure and deployment practices. Modern AI deployment
pipelines rely on containerization technologies such as Docker and Kubernetes to manage
the lifecycle of model serving instances. These systems allow engineering teams to scale
inference capacity dynamically in response to traffic patterns, reducing both latency and
operational cost.

Model versioning and experiment tracking are essential components of the machine learning
operations, or MLOps, discipline. Tools such as MLflow, Weights and Biases, and DVC provide
infrastructure for logging hyperparameters, metrics, and artifacts associated with each
training run, making it possible to reproduce past results and systematically compare
different modeling approaches. Continuous integration and continuous delivery pipelines
further automate the process of validating and deploying model updates, reducing the time
from experimentation to production.

Monitoring production AI systems is a complex and ongoing challenge. Unlike traditional
software, which tends to fail in predictable, deterministic ways, AI systems can degrade
gradually as the statistical distribution of incoming data shifts away from the training
distribution, a phenomenon known as data drift or concept drift. Detecting and responding
to such drift requires sophisticated monitoring infrastructure that tracks not just system
health metrics but also the statistical properties of model inputs and outputs over time.

Interpretability and debugging tools for deep learning models have improved substantially
in recent years, but remain far from adequate for many applications. Saliency maps, attention
visualizations, and concept activation vectors can provide partial insight into model
reasoning, but they are often noisy, inconsistent, and difficult to interpret. The development
of more reliable and actionable interpretability methods is an active area of research with
significant practical importance.

The intersection of AI with creative domains such as art, music, and writing has generated
both excitement and controversy. Generative models can now produce images, audio, and text
of impressive quality, raising questions about authorship, originality, and the future of
creative professions. Some artists and writers see AI as a powerful collaborative tool that
can augment their creative capabilities, while others view it as a threat to the livelihoods
and cultural value of human creative work.

In the physical sciences, AI is accelerating the pace of discovery. Protein structure
prediction systems have revolutionized structural biology, enabling researchers to predict
the three-dimensional structures of proteins from their amino acid sequences with remarkable
accuracy. Drug discovery pipelines now incorporate deep learning models that can predict
the binding affinity of candidate molecules for biological targets, dramatically reducing
the time and cost required to identify promising leads.

Climate science and environmental monitoring represent another frontier for AI applications.
Machine learning models can identify patterns in satellite imagery, sensor networks, and
climate simulations that would be too complex for human analysts to detect unaided. These
capabilities are being applied to tasks ranging from deforestation monitoring to extreme
weather prediction to the optimization of renewable energy systems.

In conclusion, the trajectory of AI development over the coming years will be shaped by
the interplay of technical advances, economic incentives, regulatory frameworks, and
societal values. Navigating this complex landscape wisely requires sustained
multidisciplinary engagement from researchers, policymakers, educators, and members of
the public alike. The stakes for human flourishing and for the kind of future we create
together could not be higher. Thoughtful stewardship of these powerful technologies is
among the most important collective responsibilities of our time, and the choices we make
in this critical period will reverberate for generations to come.
"""

doc_text = FILLER_BEFORE.strip()
word_count = len(doc_text.split())
print(f"Document word count: {word_count}")

# Verify NEEDLE position
words = doc_text.split()
needle_idx = None
for i, w in enumerate(words):
    if "NEEDLE:" in w:
        needle_idx = i
        break
print(f"NEEDLE position: word {needle_idx} of {word_count}")
print(f"Context window: 1200 words — needle is {'PAST' if needle_idx and needle_idx > 1200 else 'WITHIN'} truncation point")

# Verify tokenizer compatibility
key = "launchkey"
passes = key.isalpha() or key.isalnum()
print(f"Key '{key}' passes VectorStore tokenizer: {passes}")

# Save the document
doc_path = DEMO_DIR / "sample_needle_doc.txt"
doc_path.write_text(doc_text, encoding="utf-8")

# Create the task record
task = {
    "id": "demo-needle-phoenix",
    "type": "needle",
    "doc_path": str(doc_path),
    "key": "launchkey",
    "answer": "PHOENIX-7749",
    "length_words": word_count
}

output_path = DEMO_DIR / "sample_task.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(task) + "\n")

print(f"\nTask written to: {output_path}")
