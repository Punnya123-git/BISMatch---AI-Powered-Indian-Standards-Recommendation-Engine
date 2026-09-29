# BISMatch

### AI-Powered Indian Standards Recommendation Engine

BISMatch is an AI-powered web application that analyzes procurement requirements and identifies applicable Indian Standards using semantic retrieval, RAG, and LLM-based reasoning.

The system is designed for procurement specifications where the user may describe a product, equipment, technical requirement, or procurement need in natural language.

---

## What BISMatch Does

BISMatch takes a procurement requirement as input and:

- Understands the procurement requirement and technical specifications
- Extracts relevant technical attributes from the query
- Performs semantic retrieval using embeddings and ChromaDB
- Retrieves candidate Indian Standards from the verified local catalogue
- Uses an LLM to perform the final applicability reasoning
- Distinguishes between directly applicable, related/supporting, and excluded standards
- Considers standard scope, status, revision, amendments, references, and certification information when available
- Avoids recommending standards that are not supported by the retrieved catalogue evidence
- Can return no recommendation when the available evidence is insufficient
- Clearly separates recommendations from retrieved candidates and catalogue coverage gaps

---

## Architecture

```text
User Procurement Requirement
            |
            v
    React + Vite Frontend
            |
            v
       FastAPI Backend
            |
            v
   Requirement Understanding
            |
            v
 Technical Attribute Extraction
            |
            v
      Semantic Retrieval
            |
            v
   Local ONNX MiniLM Embeddings
            |
            v
         ChromaDB
            |
            v
   Candidate Standards + Evidence
            |
            v
      LLM Reasoning (Groq)
            |
            v
 FINAL Applicability Decision
            |
            v
       BISMatch Results

       Technology Stack
Frontend
React
Vite
JavaScript
CSS
Backend
Python
FastAPI
Pydantic
RAG / Retrieval
ChromaDB
Local ONNX embeddings
all-MiniLM-L6-v2
384-dimensional embeddings
Semantic similarity search
Technical attribute extraction
LLM
Groq API
OpenAI-compatible API interface
Configurable LLM provider abstraction
AI Reasoning Approach

BISMatch uses a two-stage approach.

1. Candidate Generation

The retrieval layer uses semantic search and technical signals to identify potentially relevant standards from the verified catalogue.

This stage is used only for:

Candidate generation
Evidence retrieval
Technical matching
Search support

It does not make the final applicability decision.

2. AI Applicability Reasoning

The retrieved candidates and their supporting evidence are passed to the LLM.

The LLM evaluates the procurement requirement against the retrieved standard scopes and evidence and makes the final applicability decision.

The AI can:

Select applicable standards
Identify related/supporting standards
Exclude irrelevant candidates
Return no recommendation when evidence is insufficient

The system does not use deterministic ranking as a hidden fallback for final recommendations when AI reasoning is unavailable.

Grounding and Reliability

BISMatch is designed to keep AI recommendations grounded in the available standards catalogue.

The system:

Restricts recommendations to verified catalogue standards
Prevents invented standard numbers and titles
Does not infer unstated components merely because they may exist in a facility
Requires applicability to be supported by retrieved standard evidence
Treats AI failures separately from catalogue coverage gaps
Does not present deterministic retrieval results as final AI recommendations

If the LLM is unavailable because of a provider error, rate limit, timeout, malformed response, or truncated response, BISMatch reports that AI reasoning is unavailable rather than silently presenting deterministic candidates as recommendations.

Current Standards Catalogue

The current prototype contains a curated BIS-based local catalogue of 20 Indian Standards.

The catalogue includes fields such as:

Standard number
Title
Scope
Product category
Status
Revision/year
Amendments
Related standards
References
Certification information
Source information

The current 20-standard catalogue is a prototype dataset and is not the complete BIS standards database.

BISMatch does not claim live or complete BIS database coverage.

For authoritative and current standards information, users should verify results with the official BIS resources.

Embeddings

BISMatch uses the local:

all-MiniLM-L6-v2

embedding model through ONNX.

This allows semantic retrieval without requiring an embedding API key.

The embeddings are stored and searched using ChromaDB.

LLM Configuration

The current LLM provider uses Groq through an OpenAI-compatible API.

Example configuration:

LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=3800

Never commit the real API key to GitHub.

Use environment variables for deployment.

The repository only contains example environment configuration files.

Project Structure
BISMatch/
│
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── document_processing/
│   │   ├── models/
│   │   ├── rag/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── standards/
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── data/
│   ├── standards/
│   │   ├── standards.json
│   │   └── standards_enriched_bis_verified.json
│   └── vector_store/
│
├── docs/
│   └── architecture.md
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── .env.example
│
├── .gitignore
└── README.md
Running Locally
Backend

From the project root:

cd backend

Activate the virtual environment and install dependencies:

pip install -r requirements.txt

Start the FastAPI server:

python -m uvicorn app.main:app --reload

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs
Frontend

Open another terminal:

cd frontend
npm install
npm run dev

The Vite development server will provide the frontend URL.

Testing

The backend includes tests covering:

AI reasoning
LLM reliability
Recommendation service
Semantic retrieval
ChromaDB integration
Embedding providers
Standards dataset validation
Standards APIs
Document processing
Health APIs

Run the backend test suite with:

cd backend
pytest
Important Scope Limitation

BISMatch is an academic/SIH prototype.

The current implementation uses a curated local BIS-based standards catalogue rather than a complete live BIS database.

Therefore:

A standard not returned by BISMatch does not mean that no such Indian Standard exists.

Users should verify important procurement decisions against authoritative BIS sources.

BISMatch is an independent project and is not endorsed by or affiliated with BIS.

Project Goal

The goal of BISMatch is to make Indian Standards discovery more accessible during procurement specification preparation by combining:

Natural Language Understanding + Semantic Search + RAG + AI Reasoning

instead of relying only on exact keyword matching.