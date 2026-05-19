"""
SAP-FOLK Demo: Existing Curriculum KG

Run:
    pip install streamlit neo4j openai python-dotenv "pydantic>=2"
    streamlit run SAP-FOLK_app.py

Example .env:
    OPENAI_API_KEY=your_openai_key
    NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    OPENAI_MODEL=gpt-5.5
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError



# Environment
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")


def require_env() -> None:
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not NEO4J_URI:
        missing.append("NEO4J_URI")
    if not NEO4J_USER:
        missing.append("NEO4J_USER")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if missing:
        st.error(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\n\nCreate a local .env file or export these variables before running the app."
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )



# Pydantic data models

class PlannedTarget(BaseModel):
    label: str
    property: str
    reason: str


class PlannerOutput(BaseModel):
    task_summary: str
    targets: List[PlannedTarget] = Field(default_factory=list)
    observer_prompt: str
    max_nodes_per_target: int = 25


class NodeTextItem(BaseModel):
    node_id: str
    labels: List[str]
    property: str
    text: str
    original_value_type: str = "str"


class NodeBestPractice(BaseModel):
    node_id: str
    labels: List[str]
    property: str
    source_text_excerpt: str
    best_practices: List[str] = Field(default_factory=list)


class AggregatedBestPractices(BaseModel):
    property: str
    best_practices: List[str] = Field(default_factory=list)
    rationale: str
    executor_prompt: str


class ProposedRewrite(BaseModel):
    node_id: str
    property: str
    original_text: str
    proposed_text: str
    change_summary: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)



# General helpers


def as_dict(obj: Any) -> Dict[str, Any]:
    """
    Safely serialize a Pydantic model or dict.

    This prevents the common bug:
        target.model_dump
    instead of:
        target.model_dump()
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, BaseModel):
        return obj.model_dump()

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        return dict_method()

    raise TypeError(f"Cannot convert object of type {type(obj).__name__} to dict.")


def as_json(obj: Any, indent: int = 2) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=indent)

    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=indent,
        default=str,
    )


def short_preview(text: str, limit: int = 300) -> str:
    clean = (text or "").replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def value_type_name(value: Any) -> str:
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, set):
        return "set"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def normalize_property_value_to_text(value: Any) -> str:
    """
    Converts any Neo4j property value into readable text.

    This is the important fix for StringArray values.

    Neo4j cannot do:
        toString(n.LearningOutcomes)
    if LearningOutcomes is a StringArray.

    So this app fetches:
        n.LearningOutcomes AS value

    Then converts it here.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            item_text = normalize_property_value_to_text(item)
            if item_text:
                parts.append(item_text)
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ).strip()

    return str(value).strip()


def is_non_empty_text_value(value: Any) -> bool:
    return bool(normalize_property_value_to_text(value).strip())


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Defensive JSON parser for LLM output.
    Accepts pure JSON, or extracts the first {...} object if the model adds text.
    """
    raw = (text or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError(f"Model did not return valid JSON:\n\n{raw}")



# Neo4j helpers


def run_read_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def run_write_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    driver = get_neo4j_driver()

    with driver.session() as session:
        records = session.execute_write(
            lambda tx: list(tx.run(query, params or {}))
        )
        return [record.data() for record in records]


def escape_label_or_property(value: str) -> str:
    """
    Escape backticks for safe use in dynamic Cypher label/property names.
    """
    return value.replace("`", "``")


def get_schema_snapshot() -> Dict[str, Any]:
    """
    Reads the Neo4j schema.

    First tries APOC:
        CALL apoc.meta.schema()

    Falls back to:
        CALL db.labels()
        CALL db.relationshipTypes()
        CALL db.propertyKeys()
    """
    try:
        rows = run_read_query("CALL apoc.meta.schema() YIELD value RETURN value")

        if rows:
            return {
                "source": "apoc.meta.schema",
                "schema": rows[0].get("value", {}),
            }

    except Exception as exc:
        apoc_error = str(exc)

    labels = run_read_query(
        "CALL db.labels() YIELD label RETURN collect(label) AS labels"
    )

    relationships = run_read_query(
        """
        CALL db.relationshipTypes()
        YIELD relationshipType
        RETURN collect(relationshipType) AS relationships
        """
    )

    properties = run_read_query(
        """
        CALL db.propertyKeys()
        YIELD propertyKey
        RETURN collect(propertyKey) AS properties
        """
    )

    label_names = labels[0].get("labels", []) if labels else []
    label_properties: Dict[str, List[str]] = {}

    for label in label_names:
        safe_label = escape_label_or_property(label)

        rows = run_read_query(
            f"""
            MATCH (n:`{safe_label}`)
            RETURN keys(n) AS keys
            LIMIT 100
            """
        )

        merged_keys = sorted(
            {
                key
                for row in rows
                for key in row.get("keys", [])
            }
        )

        label_properties[label] = merged_keys

    return {
        "source": "fallback_metadata",
        "schema": {
            "labels": label_names,
            "relationships": relationships[0].get("relationships", []) if relationships else [],
            "all_properties": properties[0].get("properties", []) if properties else [],
            "label_properties": label_properties,
            "apoc_error": apoc_error if "apoc_error" in locals() else None,
        },
    }


def fetch_text_nodes(label: str, prop: str, limit: int) -> List[NodeTextItem]:
    """
    Fetch node property values for observer agents.

    Important:
    This intentionally does NOT use toString(n.`property`) in Cypher,
    because Neo4j rejects toString() for StringArray values.

    Instead, it fetches the raw value and normalizes it in Python.
    """
    safe_label = escape_label_or_property(label)
    safe_prop = escape_label_or_property(prop)

    rows = run_read_query(
        f"""
        MATCH (n:`{safe_label}`)
        WHERE n.`{safe_prop}` IS NOT NULL
        RETURN
            elementId(n) AS node_id,
            labels(n) AS labels,
            n.`{safe_prop}` AS value
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )

    items: List[NodeTextItem] = []

    for row in rows:
        raw_value = row.get("value")
        text = normalize_property_value_to_text(raw_value)

        if not text:
            continue

        items.append(
            NodeTextItem(
                node_id=str(row["node_id"]),
                labels=list(row.get("labels") or []),
                property=prop,
                text=text,
                original_value_type=value_type_name(raw_value),
            )
        )

    return items


def write_node_best_practice(
    node_id: str,
    prop: str,
    practices: List[str],
) -> None:
    """
    Writes observer-level best practices to the same node.

    This does not overwrite curriculum text.
    """
    payload = {
        "property": prop,
        "best_practices": practices,
        "timestamp": int(time.time()),
    }

    run_write_query(
        """
        MATCH (n)
        WHERE elementId(n) = $node_id
        SET n.sap_folk_observed_best_practices =
            coalesce(n.sap_folk_observed_best_practices, []) + $entry,
            n.sap_folk_observed_at = datetime()
        RETURN elementId(n) AS node_id
        """,
        {
            "node_id": str(node_id),
            "entry": [json.dumps(payload, ensure_ascii=False)],
        },
    )


def write_aggregated_best_practices(
    label: str,
    prop: str,
    aggregated: AggregatedBestPractices,
) -> int:
    """
    Propagates aggregated best practices and executor prompt to all nodes
    with the relevant label.
    """
    safe_label = escape_label_or_property(label)

    rows = run_write_query(
        f"""
        MATCH (n:`{safe_label}`)
        SET n.sap_folk_aggregated_best_practices = $payload,
            n.sap_folk_aggregated_property = $property,
            n.sap_folk_executor_prompt = $executor_prompt,
            n.sap_folk_aggregated_at = datetime()
        RETURN count(n) AS updated_count
        """,
        {
            "payload": aggregated.model_dump_json(),
            "property": prop,
            "executor_prompt": aggregated.executor_prompt,
        },
    )

    return int(rows[0].get("updated_count", 0)) if rows else 0


def update_node_text(
    node_id: str,
    prop: str,
    new_text: str,
) -> None:
    """
    Writes the accepted rewrite back to Neo4j.

    Note:
    This stores the accepted rewrite as a string. If the original property was
    a list, this will replace the list with a string. That matches the current
    rewrite workflow, but review this behavior if you need to preserve arrays.
    """
    safe_prop = escape_label_or_property(prop)

    run_write_query(
        f"""
        MATCH (n)
        WHERE elementId(n) = $node_id
        SET n.`{safe_prop}` = $new_text,
            n.sap_folk_last_updated_property = $property,
            n.sap_folk_last_updated_at = datetime()
        RETURN elementId(n) AS node_id
        """,
        {
            "node_id": str(node_id),
            "property": prop,
            "new_text": new_text,
        },
    )



# OpenAI helper


def llm_json(
    client: OpenAI,
    system: str,
    user: str,
    schema_hint: str,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Calls the OpenAI Responses API and expects valid JSON.

    Includes a fallback for models/settings that do not accept temperature.
    """
    input_messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": (
                user
                + "\n\nReturn ONLY valid JSON."
                + "\nDo not include markdown fences."
                + "\nDo not include explanatory text outside the JSON."
                + "\n\nJSON schema guidance:\n"
                + schema_hint
            ),
        },
    ]

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=input_messages,
            temperature=temperature,
        )
    except TypeError:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=input_messages,
        )
    except Exception as exc:
        message = str(exc)

        if "temperature" in message.lower():
            response = client.responses.create(
                model=OPENAI_MODEL,
                input=input_messages,
            )
        else:
            raise

    text = getattr(response, "output_text", "")

    if not text:
        raise ValueError("OpenAI response did not contain output_text.")

    return extract_json_object(text)



# SAP-FOLK components


def fractal_planner(
    client: OpenAI,
    user_request: str,
    schema_snapshot: Dict[str, Any],
    max_nodes_per_target: int,
) -> PlannerOutput:
    """
    Planner:
    - Reads KG schema.
    - Identifies labels/properties relevant to the request.
    - Generates observer prompt only.
    """
    system = """
You are an expert in analyzing analyzing textual information represented in a knowledge graph.

Your task:
- Inspect a Neo4j knowledge graph schema.
- Identify node labels and textual properties that should be analyzed.
- Focus only on properties that relate directly to the user request, do not consider properties that are irrelevant or supporting.
- Generate an observer prompt that observer agents can use to identify
  reusable text-formulation best practices from each node while focusing only on identifaction not rewriting.

Constraints:
A property is relevant ONLY if the user explicitly asked to analyze that exact kind of property.

Do NOT select properties that are merely related, adjacent, contextual, supporting, or useful for background understanding.
"""

    schema_hint = """
{
  "task_summary": "Brief summary of what the will be analyzed",
  "targets": [
    {
      "label": "Neo4j label to analyze",
      "property": "Text property to analyze",
      "reason": "Why this label/property is relevant"
    }
  ],
  "observer_prompt": "Prompt for observer agents"
  }
"""

    user = f"""
User request:
{user_request}

Neo4j schema snapshot:
{json.dumps(schema_snapshot, indent=2, ensure_ascii=False, default=str)}

Maximum nodes per target:
{max_nodes_per_target}

Select no more than 8 target label/property combinations following the instructions below:
- First identify the exact text type requested by the user.
- Then select only properties whose names directly correspond to that text type.
- Do not select properties because they are semantically related.
- Do not select properties for context.
"""

    data = llm_json(
        client=client,
        system=system,
        user=user,
        schema_hint=schema_hint,
        temperature=0.1,
    )

    data["max_nodes_per_target"] = int(max_nodes_per_target)

    try:
        return PlannerOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Planner returned invalid JSON structure:\n{exc}\n\nData:\n{data}") from exc


def observer_agent(
    client: OpenAI,
    item: NodeTextItem,
    observer_prompt: str,
) -> NodeBestPractice:
    """
    Observer:
    - Reads one text property.
    - Identifies best practices already represented in it.
    - Does not rewrite the text.
    """
    system = """
You are a expert in identifying best practices within text formulation.

You analyze one node text property from a  knowledge graph.

Your task:
- Identify reusable best practices represented in the text.
- Focus on text formulation, not factual correctness.
- Do not rewrite the text.
- Do not invent facts.
- Extract practices that could be reused by similar nodes in the graph.
"""

    schema_hint = """
{
  "node_id": "neo4j-element-id",
  "labels": ["LabelA", "LabelB"],
  "property": "propertyName",
  "source_text_excerpt": "Short excerpt from the analyzed text",
  "best_practices": [
    "Reusable best practice 1",
    "Reusable best practice 2"
  ]
}
"""

    user = f"""
Task:
{observer_prompt}

Node text item:
{item.model_dump_json(indent=2)}

Identify reusable best practices from this node text.
Return best practices as concise, actionable statements.
"""

    data = llm_json(
        client=client,
        system=system,
        user=user,
        schema_hint=schema_hint,
        temperature=0.1,
    )

    data["node_id"] = item.node_id
    data["labels"] = item.labels
    data["property"] = item.property

    if not data.get("source_text_excerpt"):
        data["source_text_excerpt"] = short_preview(item.text, 500)

    try:
        return NodeBestPractice.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Observer returned invalid JSON structure:\n{exc}\n\nData:\n{data}") from exc


def results_aggregator(
    client: OpenAI,
    prop: str,
    observations: List[NodeBestPractice],
    user_request: str,
) -> AggregatedBestPractices:
    """
    Aggregator:
    - Synthesizes local best practices into a global list.
    - Generates the executor prompt based on those global best practices.
    """
    system = """
You are the expert in synthesizing best practices.

Your task:
- Analyze observer outputs from multiple information sources.
- Synthesize them into a concise, non-duplicative global list of best practices.
- Generate the executor prompt that executor agents will use to rewrite text.
- The executor prompt must instruct an agent to preserve meaning and facts while
  improving textual information based on the synthesized best practices, not merely addressing minor lingusitic errors
- The executor prompt must explicitly prohibit inventing facts.
"""

    schema_hint = """
{
  "property": "propertyName",
  "best_practices": [
    "Aggregated best practice 1",
    "Aggregated best practice 2"
  ],
  "rationale": "Brief rationale explaining the synthesized practices",
  "executor_prompt": "Full prompt to be used by executor agents"
}
"""

    observations_payload = [obs.model_dump() for obs in observations]

    user = f"""
Original user request:
{user_request}

Property being aggregated:
{prop}

Observer outputs:
{json.dumps(observations_payload, indent=2, ensure_ascii=False, default=str)}

Create:
1. A synthesized list of global best practices.
2. A complete executor prompt for rewriting this property.

The executor prompt should be specific to the property "{prop}".
"""

    data = llm_json(
        client=client,
        system=system,
        user=user,
        schema_hint=schema_hint,
        temperature=0.1,
    )

    data["property"] = prop

    try:
        return AggregatedBestPractices.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Aggregator returned invalid JSON structure:\n{exc}\n\nData:\n{data}") from exc


def executor_agent(
    client: OpenAI,
    item: NodeTextItem,
    aggregated: AggregatedBestPractices,
) -> ProposedRewrite:
    """
    Executor:
    - Uses the executor prompt generated by the Results Aggregator.
    - Proposes a rewrite.
    - Does not update Neo4j directly.
    """
    system = """
You are a an expert in text reformulation according to best practices.

You reformulate one text property from a knowledge graph.

You must:
- Use the follow the provided prompt to inmprove the formulation of textual information based on best pratices.
- Preserve the original meaning.
- Do not invent new facts.
- Do not remove important information.
- Produce a proposed rewrite for human validation.
"""

    schema_hint = """
{
  "node_id": "neo4j-element-id",
  "property": "propertyName",
  "original_text": "Original text",
  "proposed_text": "Improved text",
  "change_summary": "Brief explanation of the changes",
  "confidence": 0.85
}
"""

    user = f"""
Prompt to be followed form improving text formulation based on best practices:
{aggregated.executor_prompt}

Best practices:
{aggregated.model_dump_json(indent=2)}

Node text item:
{item.model_dump_json(indent=2)}

Propose an improved version of the text.
"""

    data = llm_json(
        client=client,
        system=system,
        user=user,
        schema_hint=schema_hint,
        temperature=0.1,
    )

    data["node_id"] = item.node_id
    data["property"] = item.property
    data["original_text"] = item.text

    if "confidence" not in data or data["confidence"] is None:
        data["confidence"] = 0.85

    try:
        return ProposedRewrite.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Executor returned invalid JSON structure:\n{exc}\n\nData:\n{data}") from exc



# Streamlit state helpers


def initialize_state() -> None:
    defaults = {
        "schema": None,
        "plan": None,
        "text_items": [],
        "observations": [],
        "aggregated_by_property": {},
        "rewrites": [],
        "accepted": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_downstream_from_plan() -> None:
    st.session_state.text_items = []
    st.session_state.observations = []
    st.session_state.aggregated_by_property = {}
    st.session_state.rewrites = []
    st.session_state.accepted = {}


def reset_downstream_from_observers() -> None:
    st.session_state.aggregated_by_property = {}
    st.session_state.rewrites = []
    st.session_state.accepted = {}


def group_observations_by_property(
    observations: List[NodeBestPractice],
) -> Dict[str, List[NodeBestPractice]]:
    grouped: Dict[str, List[NodeBestPractice]] = {}

    for obs in observations:
        grouped.setdefault(obs.property, []).append(obs)

    return grouped


def unique_labels_for_observations(
    observations: List[NodeBestPractice],
) -> List[str]:
    labels = sorted(
        {
            label
            for obs in observations
            for label in obs.labels
        }
    )

    return labels



# Streamlit app


def main() -> None:
    st.set_page_config(
        page_title="SAP-FOLK Demo: Existing Curriculum KG",
        layout="wide",
    )

    initialize_state()
    require_env()

    client = get_openai_client()

    try:
        get_neo4j_driver().verify_connectivity()
    except Exception as exc:
        st.error(f"Could not connect to Neo4j: {exc}")
        st.stop()

    st.title("SAP-FOLK Demo: Existing Curriculum KG")

    st.caption(
        "Planner → Observers → Results Aggregator → Executors → Human Validation → Neo4j Update"
    )

    with st.sidebar:
        st.header("Connection")

        st.write(f"Neo4j URI: `{NEO4J_URI}`")
        st.write(f"Neo4j user: `{NEO4J_USER}`")
        st.write(f"OpenAI model: `{OPENAI_MODEL}`")

        st.divider()

        max_nodes = st.slider(
            "Max nodes per target",
            min_value=1,
            max_value=250,
            value=30,
            step=1,
            help="Limits how many nodes are sampled for each planned label/property target.",
        )

        if st.button("Refresh Neo4j schema"):
            with st.spinner("Reading Neo4j schema..."):
                try:
                    st.session_state.schema = get_schema_snapshot()
                    st.success("Schema refreshed.")
                except Exception as exc:
                    st.error(f"Could not read schema: {exc}")

        if st.button("Clear current run"):
            st.session_state.plan = None
            reset_downstream_from_plan()
            st.success("Current run cleared.")

    user_request = st.text_area(
        "User request",
        value=(
            "Identify and propagate best practices for formulating textual information. "
        ),
        height=110,
    )

    tabs = st.tabs(
        [
            "1. Planner",
            "2. Observers",
            "3. Results Aggregator",
            "4. Executors",
            "5. Human Validation",
            "6. Write to Neo4j",
        ]
    )

    # 1. Planner

    with tabs[0]:
        st.subheader("1. Fractal Planner")

        if st.session_state.schema is None:
            with st.spinner("Reading Neo4j schema..."):
                try:
                    st.session_state.schema = get_schema_snapshot()
                except Exception as exc:
                    st.error(f"Could not read Neo4j schema: {exc}")
                    st.stop()

        with st.expander("Neo4j schema snapshot", expanded=False):
            st.json(st.session_state.schema)

        if st.button("Run Fractal Planner", type="primary"):
            with st.spinner("Planning from existing KG schema..."):
                try:
                    plan = fractal_planner(
                        client=client,
                        user_request=user_request,
                        schema_snapshot=st.session_state.schema,
                        max_nodes_per_target=max_nodes,
                    )

                    st.session_state.plan = plan
                    reset_downstream_from_plan()
                    st.success("Planner completed.")

                except Exception as exc:
                    st.error(f"Planner failed: {exc}")

        if st.session_state.plan:
            plan: PlannerOutput = st.session_state.plan

            st.markdown("### Planner summary")
            st.write(plan.task_summary)

            st.markdown("### Planned label/property targets")

            target_rows = [as_dict(target) for target in plan.targets]

            if target_rows:
                st.dataframe(
                    target_rows,
                    use_container_width=True,
                )
            else:
                st.warning("Planner returned no targets.")

            with st.expander("Observer prompt generated by planner", expanded=True):
                st.write(plan.observer_prompt)

            st.info(
                "The executor prompt is not generated by the planner. "
                "It will be generated later by the Results Aggregator."
            )

    # 2. Observers

    with tabs[1]:
        st.subheader("2. Observer Agents")

        if not st.session_state.plan:
            st.info("Run the Fractal Planner first.")
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("Fetch target nodes from existing KG"):
                    fetched: List[NodeTextItem] = []

                    for target in st.session_state.plan.targets:
                        try:
                            fetched.extend(
                                fetch_text_nodes(
                                    label=target.label,
                                    prop=target.property,
                                    limit=st.session_state.plan.max_nodes_per_target,
                                )
                            )

                        except Exception as exc:
                            st.warning(
                                f"Could not fetch `{target.label}.{target.property}`: {exc}"
                            )

                    st.session_state.text_items = fetched
                    st.session_state.observations = []
                    reset_downstream_from_observers()

                    st.success(f"Fetched {len(fetched)} node text item(s).")

            with col_b:
                run_observers_button_col, observer_status_col = st.columns(
                    [0.45, 0.55],
                    vertical_alignment="center",
                )

                with run_observers_button_col:
                    run_observers_clicked = st.button("Run Observer Agents")

                with observer_status_col:
                    observer_status = st.empty()

                if run_observers_clicked:
                    if not st.session_state.text_items:
                        st.warning("Fetch target nodes first.")
                    else:
                        observer_status.markdown("⏳ Running observer agents...")
                        observations: List[NodeBestPractice] = []
                        progress = st.progress(0)

                        for idx, item in enumerate(st.session_state.text_items):
                            try:
                                obs = observer_agent(
                                    client=client,
                                    item=item,
                                    observer_prompt=st.session_state.plan.observer_prompt,
                                )

                                observations.append(obs)

                                write_node_best_practice(
                                    node_id=obs.node_id,
                                    prop=obs.property,
                                    practices=obs.best_practices,
                                )

                            except Exception as exc:
                                st.warning(
                                    f"Observer failed for node {item.node_id}, "
                                    f"property `{item.property}`: {exc}"
                                )

                            progress.progress(
                                (idx + 1) / max(len(st.session_state.text_items), 1)
                            )

                        st.session_state.observations = observations
                        reset_downstream_from_observers()

                        observer_status.empty()

                        st.success(
                            f"Observer phase completed with {len(observations)} observation(s)."
                        )

            if st.session_state.text_items:
                st.markdown("### Text items fetched from KG")

                item_rows = [
                    {
                        "node_id": item.node_id,
                        "labels": ", ".join(item.labels),
                        "property": item.property,
                        "original_value_type": item.original_value_type,
                        "text_preview": short_preview(item.text, 300),
                    }
                    for item in st.session_state.text_items
                ]

                st.dataframe(
                    item_rows,
                    use_container_width=True,
                )

            if st.session_state.observations:
                st.markdown("### Observer outputs")

                for obs in st.session_state.observations:
                    with st.expander(f"Node {obs.node_id} · `{obs.property}`"):
                        st.markdown("**Source excerpt**")
                        st.write(obs.source_text_excerpt)

                        st.markdown("**Observed best practices**")
                        for practice in obs.best_practices:
                            st.markdown(f"- {practice}")

    # 3. Results Aggregator

    with tabs[2]:
        st.subheader("3. Results Aggregator")

        if not st.session_state.observations:
            st.info("Run Observer Agents first.")
        else:
            if st.button("Run Results Aggregator", type="primary"):
                grouped = group_observations_by_property(st.session_state.observations)
                aggregated_by_property: Dict[str, AggregatedBestPractices] = {}

                with st.spinner("Aggregating best practices and generating executor prompts..."):
                    for prop, observations in grouped.items():
                        try:
                            aggregated = results_aggregator(
                                client=client,
                                prop=prop,
                                observations=observations,
                                user_request=user_request,
                            )

                            aggregated_by_property[prop] = aggregated

                            labels_for_property = unique_labels_for_observations(observations)
                            total_updated = 0

                            for label in labels_for_property:
                                total_updated += write_aggregated_best_practices(
                                    label=label,
                                    prop=prop,
                                    aggregated=aggregated,
                                )

                            st.success(
                                f"Aggregated `{prop}` and propagated to "
                                f"{total_updated} node(s)."
                            )

                        except Exception as exc:
                            st.warning(f"Aggregation failed for `{prop}`: {exc}")

                st.session_state.aggregated_by_property = aggregated_by_property
                st.session_state.rewrites = []
                st.session_state.accepted = {}

            if st.session_state.aggregated_by_property:
                for prop, aggregated in st.session_state.aggregated_by_property.items():
                    with st.expander(
                        f"Aggregated best practices and executor prompt for `{prop}`",
                        expanded=True,
                    ):
                        st.markdown("### Rationale")
                        st.write(aggregated.rationale)

                        st.markdown("### Aggregated best practices")
                        for practice in aggregated.best_practices:
                            st.markdown(f"- {practice}")

                        st.markdown("### Executor prompt generated by Results Aggregator")
                        st.text_area(
                            label=f"Executor prompt for {prop}",
                            value=aggregated.executor_prompt,
                            height=260,
                            disabled=True,
                        )

    # 4. Executors

    with tabs[3]:
        st.subheader("4. Executor Agents")

        if not st.session_state.aggregated_by_property:
            st.info("Run the Results Aggregator first.")
        else:
            run_executors_button_col, executor_status_col = st.columns(
                [0.32, 0.68],
                vertical_alignment="center",
            )

            with run_executors_button_col:
                run_executors_clicked = st.button("Run Executor Agents", type="primary")

            with executor_status_col:
                executor_status = st.empty()

            if run_executors_clicked:
                executor_status.markdown("⏳ Running executor agents...")
                rewrites: List[ProposedRewrite] = []
                progress = st.progress(0)

                with st.spinner("Generating proposed rewrites..."):
                    for idx, item in enumerate(st.session_state.text_items):
                        aggregated = st.session_state.aggregated_by_property.get(
                            item.property
                        )

                        if not aggregated:
                            progress.progress(
                                (idx + 1) / max(len(st.session_state.text_items), 1)
                            )
                            continue

                        try:
                            rewrite = executor_agent(
                                client=client,
                                item=item,
                                aggregated=aggregated,
                            )

                            rewrites.append(rewrite)

                        except Exception as exc:
                            st.warning(
                                f"Executor failed for node {item.node_id}, "
                                f"property `{item.property}`: {exc}"
                            )

                        progress.progress(
                            (idx + 1) / max(len(st.session_state.text_items), 1)
                        )

                st.session_state.rewrites = rewrites
                st.session_state.accepted = {
                    f"{rewrite.node_id}:{rewrite.property}": False
                    for rewrite in rewrites
                }

                executor_status.empty()

                st.success(
                    f"Executor phase completed with {len(rewrites)} proposed rewrite(s)."
                )

            if st.session_state.rewrites:
                st.markdown("### Proposed rewrite summary")

                rewrite_rows = [
                    {
                        "node_id": rewrite.node_id,
                        "property": rewrite.property,
                        "confidence": rewrite.confidence,
                        "change_summary": rewrite.change_summary,
                    }
                    for rewrite in st.session_state.rewrites
                ]

                st.dataframe(
                    rewrite_rows,
                    use_container_width=True,
                )

    # 5. Human Validation

    with tabs[4]:
        st.subheader("5. Human Validation")

        if not st.session_state.rewrites:
            st.info("Run Executor Agents first.")
        else:
            col_accept, col_reject = st.columns(2)

            with col_accept:
                if st.button("Accept all proposed textual information", type="primary"):
                    for rewrite in st.session_state.rewrites:
                        key = f"{rewrite.node_id}:{rewrite.property}"
                        st.session_state.accepted[key] = True
                        st.session_state[f"accept_{key}"] = True

                    st.success("All proposed rewrites marked as accepted.")

            with col_reject:
                if st.button("Reject all proposed textual information"):
                    for rewrite in st.session_state.rewrites:
                        key = f"{rewrite.node_id}:{rewrite.property}"
                        st.session_state.accepted[key] = False
                        st.session_state[f"accept_{key}"] = False

                    st.success("All proposed rewrites marked as rejected.")

            st.divider()

            for rewrite in st.session_state.rewrites:
                key = f"{rewrite.node_id}:{rewrite.property}"

                if key not in st.session_state.accepted:
                    st.session_state.accepted[key] = False

                with st.expander(
                    f"Node {rewrite.node_id} · `{rewrite.property}` · confidence {rewrite.confidence:.2f}",
                    expanded=False,
                ):
                    st.markdown("**Change summary**")
                    st.write(rewrite.change_summary)

                    original_col, proposed_col = st.columns(2)

                    with original_col:
                        st.markdown("**Original text**")
                        st.text_area(
                            label=f"Original {key}",
                            value=rewrite.original_text,
                            height=260,
                            disabled=True,
                            label_visibility="collapsed",
                        )

                    with proposed_col:
                        st.markdown("**Proposed text**")
                        edited_text = st.text_area(
                            label=f"Proposed {key}",
                            value=rewrite.proposed_text,
                            height=260,
                            key=f"edit_{key}",
                            label_visibility="collapsed",
                        )

                        rewrite.proposed_text = edited_text

                    accepted_value = st.checkbox(
                        "Accept this rewrite",
                        value=st.session_state.accepted.get(key, False),
                        key=f"accept_{key}",
                    )

                    st.session_state.accepted[key] = accepted_value

            accepted_count = sum(
                1 for value in st.session_state.accepted.values() if value
            )

            st.success(
                f"{accepted_count} of {len(st.session_state.rewrites)} proposed rewrite(s) accepted."
            )

    # 6. Write to Neo4j

    with tabs[5]:
        st.subheader("6. Write Accepted Changes to Neo4j")

        if not st.session_state.rewrites:
            st.info("No proposed rewrites available.")
        else:
            accepted_rewrites = [
                rewrite
                for rewrite in st.session_state.rewrites
                if st.session_state.accepted.get(
                    f"{rewrite.node_id}:{rewrite.property}",
                    False,
                )
            ]

            st.write(f"Accepted rewrite(s) ready to save: **{len(accepted_rewrites)}**")

            if accepted_rewrites:
                st.warning(
                    "This will overwrite the selected text properties in the existing Neo4j KG. "
                    "Review accepted changes before continuing."
                )

                preview_rows = [
                    {
                        "node_id": rewrite.node_id,
                        "property": rewrite.property,
                        "new_text_preview": short_preview(rewrite.proposed_text, 300),
                    }
                    for rewrite in accepted_rewrites
                ]

                st.dataframe(
                    preview_rows,
                    use_container_width=True,
                )

            confirm = st.checkbox(
                "I confirm that accepted rewrites should be written to the existing Neo4j KG."
            )

            if st.button(
                "Write accepted changes to Neo4j",
                type="primary",
                disabled=not accepted_rewrites or not confirm,
            ):
                updated = 0

                for rewrite in accepted_rewrites:
                    try:
                        update_node_text(
                            node_id=rewrite.node_id,
                            prop=rewrite.property,
                            new_text=rewrite.proposed_text,
                        )

                        updated += 1

                    except Exception as exc:
                        st.error(
                            f"Failed to update node {rewrite.node_id}, "
                            f"property `{rewrite.property}`: {exc}"
                        )

                st.success(f"Updated {updated} node property value(s) in Neo4j.")


if __name__ == "__main__":
    main()
