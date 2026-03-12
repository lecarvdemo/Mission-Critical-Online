"""
Travel Planning Agent
=====================
Specialised agent that creates a detailed 3-week European travel itinerary
for December 2027.

Trip profile
------------
- Travellers : couple + 17-year-old son (3 people)
- Budget      : low-to-mid (good value, not luxurious)
- Duration    : ~3 weeks (≈ 21 nights), early-to-late December 2027
- Base cities : Prague (Czech Republic), Lapland (Finland / Swedish Lapland),
                plus a skiing destination in between
- Interests   : tourist attractions, beautiful scenery, history, wine, skiing
- Home base   : Canberra, Australia
"""

from __future__ import annotations

import os
from openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert European travel planner with deep knowledge of Central and
Northern Europe, specialising in winter travel for families.

When building an itinerary you will:
1. Suggest concrete daily activities, opening times and entry fees where relevant.
2. Recommend restaurants that are *hidden gems* – authentic, locally loved,
   great value for money.  Avoid tourist traps.  Include at least one winery or
   wine-bar suggestion per city where wine culture is present.
3. Include travel logistics between cities (train, bus, budget flights, etc.)
   with approximate costs in EUR.
4. Flag family-friendly highlights that a 17-year-old will enjoy alongside the
   adults.
5. Highlight unique winter/Christmas-market experiences at each stop.
6. Respect a *low-to-mid budget* (hostels / 3-star hotels, self-catering where
   possible, free or cheap attractions alongside paid ones).
7. Structure the output as:
     • Overview paragraph
     • Day-by-day table (Date | Location | Activities | Meals | Notes)
     • Restaurant/Food highlights section (per city)
     • Estimated daily budget breakdown (EUR per person)
8. Always include visa and entry requirements for Australian passport holders.
""".strip()


def build_user_prompt() -> str:
    return """
Please create a detailed 3-week European winter travel plan for December 2027.

Traveller profile:
- Party: couple (adults) + 1 teenager (17 years old)
- Budget style: low-price but still nice – good value, not backpacker-rough,
  not luxury
- Flying from: Canberra, Australia (via connection hub – see flight agent)
- Frequent flyer: Qantas

Destinations (in this approximate order):
1. Prague, Czech Republic – 7 nights
   • Explore the old town, castle, Christmas markets, Czech cuisine & beer,
     day-trips to Český Krumlov or Kutná Hora
2. Skiing destination (Alps or Dolomites, e.g. Innsbruck / Salzburg area or
   Slovenia's Kranjska Gora) – 4 nights
   • Family ski, scenic mountain scenery, mulled wine, cosy mountain huts
3. Lapland, Finland or Swedish Lapland (e.g. Rovaniemi or Abisko) – 7 nights
   • Authentic white Christmas, Northern Lights, reindeer sleigh, Santa Claus
     Village, snowmobile, husky safaris

Remaining nights can be used for transit/buffer or an optional extra city
(e.g. Vienna 2 nights on the way from Prague to skiing).

Please provide:
• Full day-by-day itinerary with dates (assume arrival ~1 Dec 2027, depart
  ~22 Dec 2027 to be back in Australia for Christmas)
• Restaurant recommendations (hidden gems, local favourites, good wine)
• Daily estimated budget per person in EUR
• Tips for booking ski passes, Northern Lights tours, and Christmas activities
  in advance
• Packing list tailored to sub-zero Lapland AND alpine skiing conditions
""".strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class TravelPlanningAgent:
    """Creates a detailed European winter travel plan using an LLM."""

    def __init__(self, client: AzureOpenAI | OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def run(self) -> str:
        """Invoke the LLM and return the full travel plan as a string."""
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
def create_agent() -> TravelPlanningAgent:
    """
    Build a TravelPlanningAgent from environment variables.

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

    return TravelPlanningAgent(client=client, model=model)
