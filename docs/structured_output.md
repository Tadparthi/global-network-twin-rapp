# Structured Output Pattern

This project uses Pydantic structured output for the supervisor's routing decision.
Worth understanding because it's an interview-grade pattern.

## The naive approach (what NOT to do in production)

Tell the model "respond with JSON in this format" and parse the response with regex
or `json.loads`. Looks like:

```python
prompt = "Decide the routing plan. Respond with JSON: {\"routing_plan\": [...]}"
response = llm.invoke(prompt)
match = re.search(r'\{.*\}', response.content, re.DOTALL)
parsed = json.loads(match.group())  # may fail
```

This works most of the time but fails when the model:
- Adds preamble text before the JSON
- Adds a trailing explanation after the JSON
- Returns JSON with extra keys not in your schema
- Gets the JSON formatting subtly wrong (single quotes, trailing commas)
- Hallucinates field values that don't match your expectations

You then need defensive parsing, retries, fallbacks. Bug surface area grows.

## The production pattern (used in src/agents/supervisor.py)

Define the schema as a Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import Literal

AgentName = Literal["diagnostician", "interference_analyst", "capacity_planner", "policy_writer"]

class RoutingDecision(BaseModel):
    routing_plan: list[AgentName] = Field(description="Ordered list of agents to invoke")
    reasoning: str = Field(description="One-sentence justification")
```

Bind it to the LLM via `with_structured_output()`:

```python
structured_llm = llm.with_structured_output(RoutingDecision)
decision: RoutingDecision = structured_llm.invoke([...])

# decision is now a typed Python object, guaranteed to validate.
routing_plan = decision.routing_plan  # IDE autocomplete works
```

What LangChain does behind the scenes:

1. Converts the Pydantic schema to JSON Schema
2. Includes the schema in the prompt the model sees, so the model knows the exact format expected
3. Routes the call through the LLM provider's structured output API (Anthropic's tool-use mechanism, OpenAI's response_format=json_schema, etc.)
4. Validates the response against the Pydantic schema
5. Retries with a corrective prompt if validation fails
6. Returns a typed Python object

You write less code, get type safety, and eliminate an entire class of parsing bugs.

## Why this matters in interviews

The interview question is some variant of:

> "How do you ensure the LLM returns valid structured data?"

The answer that signals you've actually built things:

> "Three approaches in increasing reliability: prompt engineering with format examples,
> Pydantic schemas via `with_structured_output()`, and tool-calling as structured
> output. I default to Pydantic structured output — LangChain handles schema conversion,
> validation, and retries. The output is a typed Python object so downstream code
> gets static type checking. The only time I'd avoid it is if I was supporting a
> model that doesn't have native structured output APIs, where I'd fall back to
> tool-calling."

That answer demonstrates:
- You know the trade-offs between approaches
- You've used the production pattern, not just read about it
- You think about failure modes and downstream consequences

## Where else to apply this pattern

Anywhere your code parses LLM output into structured data:

- Classification tasks (spam/not-spam, sentiment, category)
- Extraction tasks (entities, dates, amounts from text)
- Decision tasks (routing, prioritization, action selection)
- Generation tasks where format matters (specific JSON schemas, XML, code)

If you find yourself writing regex to parse LLM output, you're probably better off
with structured output instead.
