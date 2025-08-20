from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    security: dict | None


_SENTINEL = None


def get_detector():
    global _SENTINEL
    if _SENTINEL is None:
        model_id = "qualifire/prompt-injection-sentinel"
        kwargs = {}
        if torch.cuda.is_available():
            kwargs.update(dict(device_map="auto", torch_dtype=torch.float16))
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        _SENTINEL = pipeline(
            "text-classification", model=mdl, tokenizer=tok, top_k=None, truncation=True
        )
    return _SENTINEL


def is_prompt_injection(text: str, threshold: float = 0.80):

    raw = get_detector()(text, truncation=True)

    # --- normalize to list[dict] ---
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        preds = raw
    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
        preds = raw[0]
    else:
        preds = []

    def norm(s: str) -> str:
        return s.strip().lower().replace("-", "").replace("_", "")

    inj_labels = {"jailbreak", "promptinjection", "injection", "attack", "malicious"}
    benign_labels = {"benign", "safe", "clean"}

    inj_score = benign_score = 0.0
    top_label, top_score = None, -1.0

    for p in preds:
        lbl = norm(p.get("label", ""))
        score = float(p.get("score", 0.0))
        if lbl in inj_labels and score > inj_score:
            inj_score = score
        if lbl in benign_labels and score > benign_score:
            benign_score = score
        if score > top_score:
            top_score, top_label = score, p.get("label", "UNKNOWN")

    flagged = (inj_score >= threshold) and (inj_score >= benign_score)
    label = "jailbreak" if flagged else (top_label or "UNKNOWN")

    # debug (isteğe bağlı)
    print(
        f"[DETECT] inj={inj_score:.3f} benign={benign_score:.3f} label={label} flagged={flagged}"
    )

    return flagged, float(inj_score), label


def guard(state: State):
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) or (
            isinstance(m, dict) and m.get("role") == "user"
        ):
            last_user = m
            break
    text = last_user.content if last_user else ""
    flagged, score, label = is_prompt_injection(text, threshold=0.80)
    new_msgs = []
    sec = {"flagged": flagged, "score": score, "label": label}
    print(f"sec: {sec}")
    if flagged:
        new_msgs.append(
            AIMessage(
                content=(
                    "Bu istekte güvenlik açısından riskli bir yön algıladım. "
                    "Kuralları yok sayma veya gizli bilgileri ifşa etme talebi varsa uygulamayacağım. "
                    "Lütfen sorunu daha somut ve güvenli biçimde yeniden ifade et."
                )
            )
        )
    return {"messages": new_msgs, "security": sec}


def guard_router(state: State) -> Literal["block", "chatbot"]:
    sec = state.get("security") or {}
    flagged = bool(sec.get("flagged"))
    route = "block" if flagged else "chatbot"

    # Teşhis için:
    print(f"[ROUTER] flagged={flagged} -> {route!r} (type={type(route)})")
    assert isinstance(
        route, str
    ), f"Router must return str, got {type(route)}: {route!r}"

    return route


@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    """
    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}.get(symbol, 0.0)


tools = [get_stock_price]

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

memory = MemorySaver()
builder = StateGraph(State)
builder.add_node("guard", guard)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "guard")

builder.add_conditional_edges(
    "guard", guard_router, {"block": END, "chatbot": "chatbot"}
)

builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
graph = builder.compile(checkpointer=memory)

client = Client()


@traceable(name="call_graph", tags=["demo-bot"], metadata={"component": "agent-entry"})
def call_graph(query: str):
    st = graph.invoke({"messages": [{"role": "user", "content": query}]})
    sec = st.get("security") or {}

    run = get_current_run_tree()
    if run:
        run.add_metadata({
            "decision_source": "guard" if sec.get("flagged") else "none",
            "security_flagged": bool(sec.get("flagged")),
            "security_score": float(sec.get("score") or 0.0),
            "security_label": str(sec.get("label") or "NONE"),
        })

    msgs = st.get("messages", [])
    ai_message = [msg for msg in msgs if isinstance(msg, AIMessage) ]
    print(f"ai_message: {ai_message}")



def main(thread_id: str = "session-1"):
    print("Agent is ready. Exit: q | Reset: /reset")
    cfg = {"configurable": {"thread_id": thread_id}}

    pending_interrupt = None
    while True:
        if pending_interrupt:
            user = input(f"[onay] {pending_interrupt}\n> ").strip()
            if user.lower() in {"q","quit","exit"}: print("bye!"); break
            # Resume interrupt
            state = graph.invoke(Command(resume=user.lower()), config=cfg)
            # yazdır
            out = _last_ai_text(state.get("messages", []))
            print(f"assistant> {out}")
            pending_interrupt = state.get("__interrupt__")
            continue

        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in {"q","quit","exit"}:
            print("bye!")
            break
        if user.strip() == "/reset":
            print("Not: MemorySaver stores as in thread_id. You can use new t_id.")
            continue

        state = graph.invoke({"messages":[{"role":"user","content":user}]}, config=cfg)

        intr = state.get("__interrupt__")
        if intr:
            pending_interrupt = intr
            continue
        out = _last_ai_text(state.get("messages", []))
        print(f"assistant> {out}")



from langchain_core.messages import BaseMessage
def _last_ai_text(msgs: list) -> str:
    for m in reversed(msgs):
        is_ai = (getattr(m, "type", None) == "ai") or (isinstance(m, dict) and m.get("role")=="assistant")
        if not is_ai: continue
        c = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            c = "\n".join(parts)
        if isinstance(c, str) and c.strip():
            return c
    return "(boş yanıt)"

if __name__ == "__main__":
    main("buy_thread")
