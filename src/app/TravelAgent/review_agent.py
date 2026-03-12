"""
Review Agent
============
Specialised agent that critically reviews and challenges the outputs of the
Travel Planning Agent and the Flight Planning Agent, then synthesises a final
consolidated recommendation for the travellers.

Role
----
- Acts as an independent expert reviewer and devil's advocate
- Challenges assumptions, spots gaps, and suggests improvements
- Provides a concise, actionable final summary
"""

from __future__ import annotations

import os
from openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a senior travel and logistics consultant with 20+ years of experience
planning complex international trips, including family winter holidays in Europe.
You are known for being thorough, pragmatic and honest.

Your role is to:
1. Critically review the travel itinerary and the flight plan provided to you.
2. Challenge any unrealistic assumptions (travel times between cities, costs,
   availability of activities in winter, ski-season open dates, etc.).
3. Identify gaps or risks the family should be aware of (visa requirements,
   travel insurance edge cases, ski-pass booking deadlines, crowding at popular
   Christmas-market destinations, Northern Lights viewing probability, etc.).
4. Propose concrete improvements or alternative options where the original plans
   can be optimised for cost, comfort or experience.
5. Validate the Qantas / oneworld flight strategy and flag any issues with
   layovers, transit visas, or luggage policies for ski equipment.
6. Produce a polished, consolidated FINAL SUMMARY that a busy family can use
   as their single reference document. Structure it as:
       a. Executive Summary (5–6 sentences)
       b. Final Recommended Flight Plan (table)
       c. Final Recommended Itinerary (day-by-day table, condensed)
       d. Top Restaurant & Food Picks (one highlight per city)
       e. Key Action Items & Booking Timeline (sorted by urgency)
       f. Budget Overview (total estimated cost in AUD for 3 pax)
       g. Risks & Mitigations
       h. Final Tips for a Memorable Trip
7. Be constructive but do not simply repeat what the other agents said – add
   value through synthesis and critical insight.
""".strip()


def build_user_prompt(travel_plan: str, flight_plan: str) -> str:
    return f"""
Below are the outputs from two specialist agents.  Please review them both,
challenge where needed, and produce the final consolidated recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAVEL PLANNING AGENT OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{travel_plan}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLIGHT PLANNING AGENT OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{flight_plan}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context reminder:
- Family of 3: couple + 17-year-old son
- Budget: low-to-mid (good value, not luxury)
- Home: Canberra, Australia
- Frequent flyer: Qantas (QFF / oneworld)
- Dates: approx. 1–22 December 2027
- Destinations: Prague → skiing in Alps/Slovenia → Lapland (Finland/Sweden)
- Interests: tourist attractions, scenery, history, wine, skiing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please provide:
1. Your critical review of each agent's output (what is good, what needs
   correction or improvement)
2. The final consolidated recommendation document as described in your
   instructions above
""".strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class ReviewAgent:
    """Reviews and synthesises the outputs of the Travel and Flight agents."""

    def __init__(self, client: AzureOpenAI | OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, travel_plan: str, flight_plan: str) -> str:
        """
        Invoke the LLM with both agent outputs and return the review/summary.

        Parameters
        ----------
        travel_plan:
            Raw text output from the TravelPlanningAgent.
        flight_plan:
            Raw text output from the FlightPlanningAgent.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(travel_plan, flight_plan)},
            ],
            temperature=0.6,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def create_agent() -> ReviewAgent:
    """
    Build a ReviewAgent from environment variables.

    Azure OpenAI  (preferred when AZURE_OPENAI_ENDPOINT is set):
        AZURE_OPENAI_ENDPOINT   – e.g. https://<resource>.openai.azure.com/
        AZURE_OPENAI_API_KEY    – API key
        AZURE_OPENAI_DEPLOYMENT – deployment / model name (default: gpt-4o)
        OPENAI_API_VERSION      – API version (default: 2024-02-01)

    OpenAI fallback (when AZURE_OPENAI_ENDPOINT is *not* set):
        OPENAI_API_KEY          – OpenAI API key
        OPENAI_MODEL            – model name (default: gpt-4o)
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        client: AzureOpenAI | OpenAI = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        )
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

    return ReviewAgent(client=client, model=model)
