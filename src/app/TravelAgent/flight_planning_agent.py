"""
Flight Planning Agent
=====================
Specialised agent that researches and recommends the best flight options from
Canberra (CBR) to Europe for a December 2027 family trip.

Scope
-----
- Origin      : Canberra, Australia (CBR)
- Connection  : Sydney (SYD) or Melbourne (MEL) is assumed as the long-haul
                departure point because CBR has limited international service.
- Destinations: Prague (PRG), Rovaniemi (RVN) / Kiruna (KRN) / Kittilä (KTT)
                and an alpine skiing stop (INN / SZG / LJU)
- Travellers  : 3 (2 adults + 1 teenager, 17 years old)
- Loyalty     : Qantas Frequent Flyer (QFF)
- Budget      : economy class, value-optimised
"""

from __future__ import annotations

import os
from openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert flight planning specialist with in-depth knowledge of
international airline routes, alliances, frequent-flyer programmes – especially
Qantas Frequent Flyer (QFF) – and fare strategies.

When advising on flights you will:
1. Recommend realistic routing options from Canberra (CBR) to the European
   destinations, including layover cities and approximate flight durations.
2. Provide estimated fare ranges in AUD for economy class (lowest-available
   vs. flexible), clearly noting that prices are estimates only.
3. Give specific advice on:
   a. Best booking window (how many months in advance, day-of-week to buy)
   b. Which Qantas partner airlines operate the relevant routes (oneworld)
   c. How to use QFF points for upgrades or redemptions on these routes
   d. Baggage allowances (especially ski equipment) and associated fees
   e. Transit visa requirements for common layover hubs (e.g., Dubai, Doha,
      Singapore, London) for Australian passport holders
4. Highlight seasonal pricing patterns (December is peak; Christmas
   surcharges apply).
5. Suggest flexible-date strategies: e.g. flying out 1–2 Dec, returning
   21–22 Dec avoids Christmas premium on return.
6. Recommend travel insurance providers suitable for winter sport coverage.
7. Format the output as:
   • Executive summary (3–4 sentences)
   • Routing options table (Route | Airlines | Approx. Duration | Est. Cost AUD)
   • Booking strategy & timing section
   • QFF points optimisation section
   • Baggage & ski equipment tips
   • Insurance recommendation
""".strip()


def build_user_prompt() -> str:
    return """
Please create a comprehensive flight plan for the following trip:

Traveller details:
- Party: 2 adults + 1 teenager (17 years old) – 3 passengers in total
- Frequent flyer: Qantas Frequent Flyer (QFF), oneworld alliance
- Travel class: Economy (open to premium economy upgrade on long-haul legs
  if good value or achievable with points)
- Budget mindset: value for money, not necessarily the cheapest, but avoid
  unnecessary premiums

Trip outline:
- Depart Canberra: around 29 Nov – 1 Dec 2027
- Return to Canberra: around 22–23 Dec 2027
- Destinations in order:
    1. Prague, Czech Republic (PRG) – first stop
    2. Alpine skiing stop (Innsbruck / Salzburg / Ljubljana area)
    3. Lapland, Finland/Sweden (Rovaniemi RVN or Kittilä KTT or Kiruna KRN)
- The return leg is likely from Rovaniemi (RVN) or Helsinki (HEL) back to
  Canberra via CBR
- Internal European travel (Prague → Alps → Lapland) is planned by train /
  budget flight; focus on the intercontinental legs

Please provide:
• 2–3 routing options for the outbound leg (CBR/SYD → PRG or nearby hub)
• 1–2 routing options for the return leg (RVN/HEL → CBR/SYD)
• Estimated all-up fare per person and total for 3 pax in AUD
• Best booking window (how many months ahead, best day to purchase)
• Specific QFF earning rates or redemption options on these routes
• Tips on ski equipment transport (checked bags, rental vs. own gear)
• Travel insurance notes for a family winter-sport trip
""".strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class FlightPlanningAgent:
    """Creates a detailed flight plan using an LLM."""

    def __init__(self, client: AzureOpenAI | OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def run(self) -> str:
        """Invoke the LLM and return the full flight plan as a string."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt()},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def create_agent() -> FlightPlanningAgent:
    """
    Build a FlightPlanningAgent from environment variables.

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

    return FlightPlanningAgent(client=client, model=model)
