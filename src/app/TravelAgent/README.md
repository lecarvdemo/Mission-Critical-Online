# Travel Agent

A multi-agent pipeline that plans a 3-week European winter trip for a family of
three based in Canberra, Australia.  The system is composed of three independent
agents that run in sequence and whose outputs are fed into a reviewer.

```
┌─────────────────────────────┐
│  Travel Planning Agent      │──► itinerary, restaurants, daily budget
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Flight Planning Agent      │──► routes, fares, QFF tips, baggage advice
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Review Agent               │──► critical review + final consolidated summary
└─────────────────────────────┘
```

## Trip profile

| Field        | Details                                         |
|--------------|-------------------------------------------------|
| Travellers   | Couple + 17-year-old son (3 people)             |
| Budget       | Low-to-mid (good value, not luxury)             |
| Dates        | 1 – 22 December 2027                            |
| Home base    | Canberra, Australia (CBR)                       |
| Frequent flyer | Qantas (QFF / oneworld)                      |
| Destinations | Prague → Alpine skiing → Lapland               |
| Interests    | Tourist attractions, scenery, history, wine, skiing |

## Prerequisites

- Python 3.11+
- An **Azure OpenAI** resource _or_ an **OpenAI** API key

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the `TravelAgent` directory (or export the variables
directly to your shell):

### Azure OpenAI

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o        # or your deployment name
OPENAI_API_VERSION=2024-02-01         # optional, defaults to 2024-02-01
```

### OpenAI

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o                   # optional, defaults to gpt-4o
```

## Running the pipeline

```bash
# Print all agent outputs to the terminal
python main.py

# Save the combined report to a Markdown file
python main.py --output results/trip_december_2027.md

# Explicitly point at a .env file
python main.py --env /path/to/.env --output results/trip.md
```

The pipeline will:

1. Ask the **Travel Planning Agent** to create a detailed day-by-day itinerary
   covering Prague, a skiing stop, and Lapland, including restaurant picks and
   a daily budget.
2. Ask the **Flight Planning Agent** to recommend intercontinental routing
   options from Canberra, fare estimates in AUD, QFF earning/redemption tips,
   ski-equipment baggage advice, and insurance guidance.
3. Ask the **Review Agent** to critically evaluate both plans, challenge any
   unrealistic assumptions, and produce a single polished final summary.

## Project structure

```
TravelAgent/
├── main.py                   # Orchestrator – runs all three agents
├── travel_planning_agent.py  # Agent 1: Europe itinerary
├── flight_planning_agent.py  # Agent 2: Canberra → Europe flights
├── review_agent.py           # Agent 3: Review & final summary
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Agents

### Travel Planning Agent (`travel_planning_agent.py`)

- Specialist: European winter travel, family trips, budget travel
- Output: day-by-day itinerary, restaurant hidden gems, daily EUR budget,
  packing list, booking tips for ski passes & Northern Lights tours

### Flight Planning Agent (`flight_planning_agent.py`)

- Specialist: international routing, Qantas / oneworld, fare strategy
- Output: routing table, AUD fare estimates, QFF tips, best booking window,
  ski-equipment baggage advice, travel insurance notes

### Review Agent (`review_agent.py`)

- Specialist: critical review, risk identification, synthesis
- Input: outputs of the two agents above
- Output: critical notes on each plan + final consolidated summary

## Notes

- All LLM outputs are **generated content** and should be verified against
  current airline websites, booking platforms, and travel advisories before
  making reservations.
- Prices, visa requirements, and opening times change; always cross-check with
  official sources.
- The `.env` file should **never** be committed to source control.  It is
  listed in `.gitignore`.
