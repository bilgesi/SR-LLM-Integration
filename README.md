# SR-LLM-Integration

 **This project finds math equations for real-world physics using AI.**  
We use **Symbolic Regression (SR)** to find equations and **Large Language Models (LLMs)** to check if the equations make sense.  
If the equation is incorrect, we use LLM feedback to help SR find a better one.  

---

##  **Project Goal**
- Simulate **a ball dropping from a height** using real physics.
- Use **Symbolic Regression (PySR & gplearn)** to find the equation of motion.
- Ask **LLMs (GPT-2/Mistral/DeepSeek)** if the equation is **correct**.
- If the equation is **incorrect**, use LLM feedback to improve SR's equation.

---

##  **Research Question: "Integrating LLM into SR Optimization"**
**Symbolic Regression (SR)** finds equations that fit data, but:  
- Just because an equation fits the data **doesn't mean it follows real-world physics**.  
- We ask **LLMs** (GPT-2, Mistral) to check if the equation is **correct**.  
- If the equation is **incorrect**, we use LLM feedback to **help SR generate a better equation**.  

---

## **SR × LLM Feedback Table**
| SR Model  | Equation | LLM Used | Equation Correct? | LLM Feedback | Suggested Improvement |
|-----------|---------|----------|------------------|--------------|------------------------|
| **PySR** | `y = 25.20491 - exp(x0)` | GPT-2 | ❌ No | Response was unclear and not useful | Use `y = a*x0^2 + b*x0 + c` |
| **gplearn** | Complex equation (log, sqrt, div, etc.) | GPT-2 | ❌ No | Output was nonsensical | Simplify with polynomial functions |
| **PySR** | `y = 25.20491 - exp(x0)` | Mistral | ❌ No | Incorrect for free fall, should be quadratic | Use `y = a*x0^2 + b*x0 + c` |
| **gplearn** | Complex equation (log, sqrt, div, etc.) | Mistral | ✅ Yes | Equation is valid | No changes needed |


