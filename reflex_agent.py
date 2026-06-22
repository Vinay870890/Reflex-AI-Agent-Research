import os
import json
import math
import subprocess
import tempfile
import streamlit as st
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient

# ---------------- CONFIG ----------------

HF_TOKEN = os.getenv("reflex", "Your API KEY")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# ---------------- LLM INTERFACE ----------------

class LLMInterface:
    def __init__(self, token: str, model: str = MODEL_NAME):
        self.client = InferenceClient(model=model, token=token)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


# ---------------- TOOLS ----------------

class Calculator:
    """Evaluates safe mathematical expressions."""
    name = "Calculator"

    def run(self, expression: str) -> str:
        try:
            # Allow only safe math operations
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculator result: {result}"
        except Exception as e:
            return f"Calculator error: {str(e)}"


class CodeGenerator:
    """Generates code using the LLM."""
    name = "Code Generator"

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def run(self, task: str) -> str:
        system = (
            "You are an expert programmer. Generate clean, well-commented code. "
            "Return ONLY the code block, no extra prose."
        )
        code = self.llm.generate(system, f"Write code for: {task}", temperature=0.3)
        return f"Generated Code:\n{code}"


class CodeExecutor:
    """Executes Python code in a sandboxed subprocess."""
    name = "Code Executor"

    def run(self, code: str, timeout: int = 10) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(
                ["python3", fname],
                capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return f"Execution output:\n{output}"
        except subprocess.TimeoutExpired:
            return "Code Executor error: execution timed out."
        except Exception as e:
            return f"Code Executor error: {str(e)}"
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass


class ToolRouter:
    """Decides which tool to invoke based on the task and answer."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.calculator = Calculator()
        self.code_gen = CodeGenerator(llm)
        self.code_exec = CodeExecutor()

    def decide_tool(self, task: str) -> str:
        system = (
            "You decide which tool to use for a task. "
            "Reply with ONLY one word: calculator, codegenerator, codeexecutor, or none."
        )
        decision = self.llm.generate(system, f"Task: {task}", temperature=0.1).lower().strip()
        for keyword in ["calculator", "codegenerator", "codeexecutor", "none"]:
            if keyword in decision:
                return keyword
        return "none"

    def run_tool(self, tool_name: str, task: str, code: Optional[str] = None) -> str:
        if tool_name == "calculator":
            # Extract expression from task
            return self.calculator.run(task)
        elif tool_name == "codegenerator":
            return self.code_gen.run(task)
        elif tool_name == "codeexecutor" and code:
            return self.code_exec.run(code)
        return ""


# ---------------- SIMPLE RETRIEVER ----------------

class SimpleRetriever:
    """
    A lightweight keyword-based retriever over a small in-memory knowledge base.
    Swap this out for a real vector store (FAISS, ChromaDB, etc.) in production.
    """
    def __init__(self):
        self.docs = []

    def add_document(self, text: str):
        self.docs.append(text)

    def retrieve(self, query: str, top_k: int = 2) -> str:
        if not self.docs:
            return ""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.docs:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        relevant = [doc for score, doc in scored[:top_k] if score > 0]
        return "\n---\n".join(relevant) if relevant else ""


# ---------------- PROMPTS ----------------

PLANNER_PROMPT = """
You are the Planner of a Reflex AI Agent.

Analyze the user task and create a clear step-by-step plan.

Rules:
- Only produce a plan
- Do not solve the task
- Keep steps concise

Output:

Plan:
1. ...
2. ...
3. ...
"""

SOLVER_PROMPT = """
You are an advanced Reflex AI Agent with access to memory, external knowledge, and tools.
Your goal is to produce the best possible answer through reasoning, tool use, and self-improvement.

You are given:
TASK:
{task}

PLAN:
{plan}

LONG-TERM MEMORY (learned rules from past mistakes):
{memory}

RETRIEVED CONTEXT (external knowledge, may be empty):
{retrieved_docs}

AVAILABLE TOOLS:
1. Calculator → for math problems
2. Code Generator → for writing code
3. Code Executor → for running code

----------------------------------------

INSTRUCTIONS:
Step 1: Understand the task deeply.
Step 2: Use MEMORY:
- Apply past learned rules
- Avoid repeating previous mistakes
Step 3: Use CONTEXT:
- If context is provided, treat it as the primary source
- Do NOT hallucinate beyond it
Step 4: Decide on TOOL USAGE:
- If math → use Calculator
- If coding → use Code Generator
- If execution needed → use Code Executor
- Otherwise → solve directly
Step 5: Generate Answer:
- Be clear, structured, and accurate
- Include examples if helpful
- Keep explanation appropriate to the user level
Step 6: Self-Check (IMPORTANT):
- Check for missing steps
- Check clarity and correctness
- Improve before finalizing

----------------------------------------

OUTPUT FORMAT:

Final Answer:
<your improved answer>

(Optional)
Tool Used:
<tool name or "none">

Reasoning Summary:
<short explanation of how you solved it>
"""

CRITIC_PROMPT = """
You are a Critic AI.

Evaluate the answer based on:
1. Accuracy
2. Completeness
3. Clarity
4. Logical reasoning
5. Relevance

Return ONLY JSON:

{
 "score": number_between_1_and_10,
 "critique": "short explanation",
 "is_sufficient": true_or_false
}
"""

REFLECTION_PROMPT = """
You are a Reflection AI.

Analyze the critique and explain how the answer should be improved.

Focus on:
- Missing information
- Logical mistakes
- Clarity improvements

Be concise.
"""

TOOL_DECISION_SYSTEM = """
You are the Reflex Agent's Tool Decision module.
Given the task and the current draft answer, decide if any tool should be invoked.
Return ONLY JSON:
{
  "use_tool": true_or_false,
  "tool": "calculator" | "codegenerator" | "codeexecutor" | "none",
  "input": "exact expression or code snippet or task description to pass to the tool"
}
"""


# ---------------- AGENTS ----------------

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def create_plan(self, task: str) -> str:
        return self.llm.generate(PLANNER_PROMPT, task)


class ActorAgent:
    def __init__(self, llm):
        self.llm = llm

    def _build_prompt(self, task, plan, memory, retrieved_docs):
        return SOLVER_PROMPT.format(
            task=task,
            plan=plan,
            memory=memory or "No previous reflections.",
            retrieved_docs=retrieved_docs or "No external context available."
        )

    def generate_answer(self, task: str, plan: str, memory: str = "", retrieved_docs: str = "") -> str:
        prompt = self._build_prompt(task, plan, memory, retrieved_docs)
        return self.llm.generate("You are an advanced Reflex AI Agent.", prompt)

    def refine_answer(self, task: str, previous_answer: str, reflection: str,
                      memory: str = "", retrieved_docs: str = "") -> str:
        prompt = f"""
Task:
{task}

Previous Answer:
{previous_answer}

Improvement Guidance:
{reflection}

Long-Term Memory:
{memory or "No previous reflections."}

Retrieved Context:
{retrieved_docs or "No external context available."}

Write an improved answer following the same structured output format.
"""
        return self.llm.generate("You are an advanced Reflex AI Agent.", prompt)


class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, task: str, answer: str) -> Dict[str, Any]:
        user_prompt = f"Task:\n{task}\n\nAnswer:\n{answer}"
        raw = self.llm.generate(CRITIC_PROMPT, user_prompt, temperature=0.1)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            data["score"] = int(data.get("score", 5))
            return data
        except Exception as e:
            return {"score": 5, "critique": f"Parsing failed: {str(e)}", "is_sufficient": False}


class ReflectionAgent:
    def __init__(self, llm):
        self.llm = llm

    def reflect(self, task: str, answer: str, critique: str) -> str:
        prompt = f"Task:\n{task}\n\nAnswer:\n{answer}\n\nCritique:\n{critique}"
        return self.llm.generate(REFLECTION_PROMPT, prompt)


# ---------------- MEMORY ----------------

class Memory:
    def __init__(self):
        self.reflections = []

    def add(self, reflection: str):
        self.reflections.append(reflection)

    def get_memory(self) -> str:
        if not self.reflections:
            return "No previous reflections."
        return "\n".join(f"- {r}" for r in self.reflections)


# ---------------- CONTROLLER ----------------

class ReflexAgentController:
    def __init__(self, token: str, max_iterations: int = 3):
        self.llm = LLMInterface(token)
        self.planner = PlannerAgent(self.llm)
        self.actor = ActorAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm)
        self.reflector = ReflectionAgent(self.llm)
        self.memory = Memory()
        self.tool_router = ToolRouter(self.llm)
        self.retriever = SimpleRetriever()
        self.max_iterations = max_iterations

    def _decide_and_run_tool(self, task: str, answer: str) -> tuple[str, str]:
        """Ask LLM if a tool should be used; run it if so. Returns (tool_name, tool_output)."""
        raw = self.llm.generate(
            TOOL_DECISION_SYSTEM,
            f"Task:\n{task}\n\nCurrent Answer:\n{answer}",
            temperature=0.1
        )
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            decision = json.loads(raw[start:end])
            if decision.get("use_tool") and decision.get("tool", "none") != "none":
                tool_name = decision["tool"]
                tool_input = decision.get("input", task)
                tool_output = self.tool_router.run_tool(tool_name, tool_input, code=tool_input)
                return tool_name, tool_output
        except Exception:
            pass
        return "none", ""

    def run(self, task: str):
        logs = []
        iteration_outputs = []

        # Retrieve external context
        retrieved_docs = self.retriever.retrieve(task)

        # Planning
        plan = self.planner.create_plan(task)
        logs.append(("plan", "Plan generated ✅"))

        # Initial answer
        current_answer = self.actor.generate_answer(
            task, plan,
            memory=self.memory.get_memory(),
            retrieved_docs=retrieved_docs
        )

        # Tool usage on initial answer
        tool_name, tool_output = self._decide_and_run_tool(task, current_answer)
        if tool_output:
            logs.append(("tool", f"Tool used: **{tool_name}** → {tool_output[:200]}"))
            current_answer += f"\n\n[Tool: {tool_name}]\n{tool_output}"

        for i in range(1, self.max_iterations + 1):
            logs.append(("iter", f"Iteration {i} started"))

            iteration_outputs.append({"iteration": i, "answer": current_answer, "tool": tool_name})

            eval_data = self.evaluator.evaluate(task, current_answer)
            score = eval_data["score"]
            logs.append(("score", f"Score: {score}/10 — {eval_data['critique']}"))

            if score >= 8 or eval_data["is_sufficient"]:
                logs.append(("done", "✅ Quality threshold reached"))
                break

            reflection = self.reflector.reflect(task, current_answer, eval_data["critique"])
            self.memory.add(reflection)
            logs.append(("reflect", f"Reflection: {reflection[:150]}..."))

            if i < self.max_iterations:
                current_answer = self.actor.refine_answer(
                    task, current_answer, reflection,
                    memory=self.memory.get_memory(),
                    retrieved_docs=retrieved_docs
                )
                # Tool usage after refinement
                tool_name, tool_output = self._decide_and_run_tool(task, current_answer)
                if tool_output:
                    logs.append(("tool", f"Tool used: **{tool_name}** → {tool_output[:200]}"))
                    current_answer += f"\n\n[Tool: {tool_name}]\n{tool_output}"

        return plan, current_answer, logs, iteration_outputs


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Reflex AI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 15px; }
    .log-plan   { background:#1e3a5f; color:#90caf9; padding:6px 10px; border-radius:6px; margin:3px 0; }
    .log-iter   { background:#1a3324; color:#a5d6a7; padding:6px 10px; border-radius:6px; margin:3px 0; }
    .log-score  { background:#3e2723; color:#ffcc80; padding:6px 10px; border-radius:6px; margin:3px 0; }
    .log-reflect{ background:#311b5e; color:#ce93d8; padding:6px 10px; border-radius:6px; margin:3px 0; }
    .log-tool   { background:#003333; color:#80cbc4; padding:6px 10px; border-radius:6px; margin:3px 0; }
    .log-done   { background:#1b5e20; color:#c8e6c9; padding:6px 10px; border-radius:6px; margin:3px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Reflex AI Agent Dashboard")
st.caption("Memory · Tool Use · Retrieval · Self-Improvement")

col1, col2 = st.columns([2, 1])

with col1:
    task = st.text_area("Enter your task / prompt", height=130, placeholder="e.g. Explain recursion with a Python example, or Calculate 23 * 47 + sqrt(144)")

with col2:
    max_iter = st.slider("Max Iterations", 1, 5, 3)
    st.markdown("**Add Knowledge (optional)**")
    knowledge = st.text_area("Paste external context / docs", height=80,
                              placeholder="Paste any reference text to help the agent answer…")

run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True)

if run_btn and task.strip():
    controller = ReflexAgentController(token=HF_TOKEN, max_iterations=max_iter)

    if knowledge.strip():
        controller.retriever.add_document(knowledge.strip())

    with st.spinner("Reflex Agent is thinking..."):
        plan, result, logs, iteration_outputs = controller.run(task)

    # --- Plan ---
    st.subheader("🧠 Plan")
    st.write(plan)

    # --- Logs ---
    st.subheader("📋 Agent Logs")
    type_class = {"plan": "log-plan", "iter": "log-iter", "score": "log-score",
                  "reflect": "log-reflect", "tool": "log-tool", "done": "log-done"}
    for log_type, log_msg in logs:
        css = type_class.get(log_type, "log-iter")
        st.markdown(f'<div class="{css}">{log_msg}</div>', unsafe_allow_html=True)

    # --- Iterations ---
    st.subheader("🔁 Iteration Outputs")
    for item in iteration_outputs:
        label = f"Iteration {item['iteration']}"
        if item.get("tool") and item["tool"] != "none":
            label += f"  🔧 Tool: {item['tool']}"
        with st.expander(label):
            st.write(item["answer"])

    # --- Final ---
    st.subheader("✅ Final Output")
    st.success(result)

elif run_btn:
    st.warning("Please enter a task before running the agent.")
