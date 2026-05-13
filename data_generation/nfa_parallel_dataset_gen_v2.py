# """
# Parallel NFA Dataset Generator v2 — 5-Tier Difficulty
# =======================================================
# Generates TWO parallel datasets from the same regex instances:
#   1. CoT   dataset : question + Thompson NFA steps + final NFA table
#   2. NoCoT dataset : question + final NFA table ONLY

# 5 tiers by NFA state count (complexity proxy):
#   Tier 1 : 4-8   states  — concat + union only             (trivial)
#   Tier 2 : 8-14  states  — + star/plus/optional            (easy)
#   Tier 3 : 14-22 states  — + star_union/star_concat        (medium)
#   Tier 4 : 22-32 states  — + nested closures               (hard)
#   Tier 5 : 32-50 states  — + deeply nested/chained ops     (very hard)

# Test set is stratified: equal examples per tier so difficulty is controlled.

# Output files (JSONL, Qwen ChatML):
#   nfa_v2_cot_train.jsonl / nfa_v2_cot_val.jsonl / nfa_v2_cot_test.jsonl
#   nfa_v2_nocot_train.jsonl / nfa_v2_nocot_val.jsonl / nfa_v2_nocot_test.jsonl
#   nfa_v2_cot_full.jsonl / nfa_v2_nocot_full.jsonl
# """

# import random
# import json
# import hashlib
# import itertools
# import argparse
# from collections import defaultdict, deque
# from copy import deepcopy
# from typing import Optional

# try:
#     from tqdm import tqdm
# except ImportError:
#     class tqdm:
#         def __init__(self, total=None, **kwargs): self.n = 0; self.total = total
#         def update(self, n=1):
#             self.n += n
#             print(f"\r  {self.n}/{self.total}", end="", flush=True)
#         def __enter__(self): return self
#         def __exit__(self, *a): print()


# # ===================
# #  1.  REGEX AST
# # ===================

# class RegexNode: pass
# class Literal(RegexNode):
#     def __init__(self, char): self.char = char
# class Epsilon(RegexNode): pass
# class Concat(RegexNode):
#     def __init__(self, l, r): self.left = l; self.right = r
# class Union(RegexNode):
#     def __init__(self, l, r): self.left = l; self.right = r
# class Star(RegexNode):
#     def __init__(self, c): self.child = c
# class Plus(RegexNode):
#     def __init__(self, c): self.child = c
# class Optional_(RegexNode):
#     def __init__(self, c): self.child = c


# def desugar(node):
#     if isinstance(node, (Literal, Epsilon)): return node
#     if isinstance(node, Concat):   return Concat(desugar(node.left), desugar(node.right))
#     if isinstance(node, Union):    return Union(desugar(node.left), desugar(node.right))
#     if isinstance(node, Star):     return Star(desugar(node.child))
#     if isinstance(node, Plus):
#         d = desugar(node.child); return Concat(d, Star(deepcopy(d)))
#     if isinstance(node, Optional_):return Union(desugar(node.child), Epsilon())
#     raise ValueError(f"Unknown node {node}")


# def regex_to_string(node, parent_prec=0):
#     if isinstance(node, Literal):   return node.char
#     if isinstance(node, Epsilon):   return "eps"
#     if isinstance(node, Star):
#         inner = regex_to_string(node.child, 3)
#         return f"({inner})*" if isinstance(node.child, (Concat, Union)) else f"{inner}*"
#     if isinstance(node, Plus):
#         inner = regex_to_string(node.child, 3)
#         return f"({inner})+" if isinstance(node.child, (Concat, Union)) else f"{inner}+"
#     if isinstance(node, Optional_):
#         inner = regex_to_string(node.child, 3)
#         return f"({inner})?" if isinstance(node.child, (Concat, Union)) else f"{inner}?"
#     if isinstance(node, Concat):
#         l = regex_to_string(node.left, 2); r = regex_to_string(node.right, 2)
#         s = l + r; return f"({s})" if parent_prec > 2 else s
#     if isinstance(node, Union):
#         l = regex_to_string(node.left, 1); r = regex_to_string(node.right, 1)
#         s = f"{l}|{r}"; return f"({s})" if parent_prec > 1 else s
#     return "?"


# # ===================
# #  2.  ALPHABETS
# # ===================

# ALPHABETS = {
#     "binary": ["0", "1"],
#     "ab":     ["a", "b"],
#     "abc":    ["a", "b", "c"],
#     "abcd":   ["a", "b", "c", "d"],
#     "digits": ["0", "1", "2", "3"],
# }

# # ===================
# #  3.  5-TIER RANDOM REGEX GENERATOR
# # ===================

# # Per-tier configuration
# TIER_CONFIG = {
#     1: {
#         "max_depth": 3,
#         "base_leaf": 0.55,
#         "ops": ["concat", "union", "concat3"],
#         "weights": [0.55, 0.35, 0.10],
#         "nfa_min": 4, "nfa_max": 8,
#         "desc": "concat+union only",
#     },
#     2: {
#         "max_depth": 4,
#         "base_leaf": 0.42,
#         "ops": ["concat", "union", "star", "plus", "optional", "concat3"],
#         "weights": [0.28, 0.20, 0.22, 0.14, 0.08, 0.08],
#         "nfa_min": 8, "nfa_max": 14,
#         "desc": "closures introduced",
#     },
#     3: {
#         "max_depth": 5,
#         "base_leaf": 0.30,
#         "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat"],
#         "weights": [0.22, 0.16, 0.20, 0.12, 0.10, 0.12, 0.08],
#         "nfa_min": 14, "nfa_max": 22,
#         "desc": "star_union/star_concat",
#     },
#     4: {
#         "max_depth": 6,
#         "base_leaf": 0.22,
#         "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat", "concat4"],
#         "weights": [0.18, 0.14, 0.20, 0.10, 0.10, 0.12, 0.08, 0.08],
#         "nfa_min": 22, "nfa_max": 32,
#         "desc": "nested closures + concat4",
#     },
#     5: {
#         "max_depth": 8,
#         "base_leaf": 0.15,
#         "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat",
#                 "concat4", "star_star", "union_chain"],
#         "weights": [0.14, 0.10, 0.18, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08, 0.06],
#         "nfa_min": 32, "nfa_max": 50,
#         "desc": "deeply nested/chained",
#     },
# }

# # Tier distribution for training set
# TIER_DIST = {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.20}


# def random_regex(tier: int, alphabet: list, rng: random.Random, depth: int = 0) -> RegexNode:
#     cfg       = TIER_CONFIG[tier]
#     max_depth = cfg["max_depth"]
#     base_leaf = cfg["base_leaf"]
#     leaf_prob = min(0.95, base_leaf + 0.13 * depth)

#     if depth >= max_depth or rng.random() < leaf_prob:
#         return Literal(rng.choice(alphabet))

#     op = rng.choices(cfg["ops"], weights=cfg["weights"])[0]

#     def sub(d=1): return random_regex(tier, alphabet, rng, depth + d)

#     if op == "concat":      return Concat(sub(), sub())
#     if op == "concat3":     return Concat(Concat(sub(), sub()), sub())
#     if op == "concat4":     return Concat(Concat(Concat(sub(), sub()), sub()), sub())
#     if op == "union":       return Union(sub(), sub())
#     if op == "union_chain": return Union(Union(sub(), sub()), sub())
#     if op == "star":        return Star(sub())
#     if op == "plus":        return Plus(sub())
#     if op == "optional":    return Optional_(sub())
#     if op == "star_union":  return Star(Union(sub(), sub()))
#     if op == "star_concat": return Star(Concat(sub(), sub()))
#     if op == "star_star":   return Star(Star(sub()))  # (r*)* = r* but produces complex NFA
#     return Literal(rng.choice(alphabet))


# # ===================
# #  4.  THOMPSON'S NFA
# # ===================

# class NFAState:
#     _counter = 0
#     def __init__(self):
#         NFAState._counter += 1
#         self.id = NFAState._counter
#         self.transitions  = defaultdict(list)
#         self.epsilon_transitions = []
#     def __repr__(self):  return f"q{self.id}"
#     def __hash__(self):  return hash(self.id)
#     def __eq__(self, o): return isinstance(o, NFAState) and self.id == o.id


# class NFA:
#     def __init__(self, start, accept):
#         self.start  = start
#         self.accept = accept

#     def all_states(self):
#         visited, stack = set(), [self.start]
#         while stack:
#             s = stack.pop()
#             if s in visited: continue
#             visited.add(s)
#             for tgts in s.transitions.values():
#                 stack.extend(t for t in tgts if t not in visited)
#             stack.extend(t for t in s.epsilon_transitions if t not in visited)
#         return visited


# def thompson(node) -> NFA:
#     if isinstance(node, Literal):
#         s, a = NFAState(), NFAState()
#         s.transitions[node.char].append(a)
#         return NFA(s, a)
#     if isinstance(node, Epsilon):
#         s, a = NFAState(), NFAState()
#         s.epsilon_transitions.append(a)
#         return NFA(s, a)
#     if isinstance(node, Concat):
#         n1 = thompson(node.left); n2 = thompson(node.right)
#         n1.accept.epsilon_transitions.append(n2.start)
#         return NFA(n1.start, n2.accept)
#     if isinstance(node, Union):
#         n1 = thompson(node.left); n2 = thompson(node.right)
#         s, a = NFAState(), NFAState()
#         s.epsilon_transitions.extend([n1.start, n2.start])
#         n1.accept.epsilon_transitions.append(a)
#         n2.accept.epsilon_transitions.append(a)
#         return NFA(s, a)
#     if isinstance(node, Star):
#         n = thompson(node.child)
#         s, a = NFAState(), NFAState()
#         s.epsilon_transitions.extend([n.start, a])
#         n.accept.epsilon_transitions.extend([n.start, a])
#         return NFA(s, a)
#     raise ValueError(f"thompson: unexpected {type(node)}")


# # ===================
# #  5.  VERIFICATION
# # ===================

# def epsilon_closure(states):
#     stack, closure = list(states), set(states)
#     while stack:
#         s = stack.pop()
#         for t in s.epsilon_transitions:
#             if t not in closure:
#                 closure.add(t); stack.append(t)
#     return frozenset(closure)


# def nfa_move(states, char):
#     result = set()
#     for s in states:
#         result.update(s.transitions.get(char, []))
#     return frozenset(result)


# def nfa_accepts(nfa, s):
#     states = epsilon_closure(frozenset([nfa.start]))
#     for c in s:
#         states = epsilon_closure(nfa_move(states, c))
#     return nfa.accept in states


# def build_dfa_for_verify(nfa, alphabet):
#     """Minimal DFA build just for cross-checking — not exposed in dataset."""
#     from collections import deque as dq
#     class DFA:
#         def __init__(self): self.transitions={}; self.start=0; self.accept_states=set(); self.states=[]
#         def accepts(self, s):
#             cur = self.start
#             for c in s:
#                 if (cur,c) not in self.transitions: return False
#                 cur = self.transitions[(cur,c)]
#             return cur in self.accept_states
#     dfa = DFA()
#     sc = epsilon_closure(frozenset([nfa.start]))
#     dfa.states.append(sc); idx={sc:0}
#     if nfa.accept in sc: dfa.accept_states.add(0)
#     q = dq([sc])
#     while q:
#         cur = q.popleft(); ci = idx[cur]
#         for c in sorted(alphabet):
#             nxt = epsilon_closure(nfa_move(cur, c))
#             if not nxt: continue
#             if nxt not in idx:
#                 i = len(dfa.states); dfa.states.append(nxt); idx[nxt]=i
#                 if nfa.accept in nxt: dfa.accept_states.add(i)
#                 q.append(nxt)
#             dfa.transitions[(ci,c)] = idx[nxt]
#     return dfa


# def verify(nfa, dfa, alphabet, max_len=5):
#     for length in range(max_len+1):
#         for combo in itertools.product(alphabet, repeat=length):
#             s = "".join(combo)
#             if nfa_accepts(nfa, s) != dfa.accepts(s):
#                 return False
#     return True


# # ===================
# #  6.  NFA TABLE FORMATTER
# # ===================

# def fmt_q(s): return f"q{s.id}"

# def fmt_nfa_set(fs):
#     ids = sorted(s.id for s in fs)
#     return "{" + ", ".join(f"q{i}" for i in ids) + "}"


# def build_nfa_table(nfa, alphabet):
#     sa = sorted(alphabet)
#     states = sorted(nfa.all_states(), key=lambda s: s.id)
#     header = "| State | Role | " + " | ".join(sa) + " | ε (epsilon) |"
#     sep    = "|-------|------|" + "|".join(["--------"]*len(sa)) + "|-------------|"
#     lines  = [header, sep]
#     for s in states:
#         role = ""
#         if s == nfa.start and s == nfa.accept: role = "start, accept"
#         elif s == nfa.start:  role = "start"
#         elif s == nfa.accept: role = "accept"
#         sym_cols = []
#         for c in sa:
#             tgts = s.transitions.get(c, [])
#             sym_cols.append("{" + ", ".join(fmt_q(t) for t in sorted(tgts, key=lambda x: x.id)) + "}"
#                             if tgts else "∅")
#         eps_tgts = s.epsilon_transitions
#         eps_cell = ("{" + ", ".join(fmt_q(t) for t in sorted(eps_tgts, key=lambda x: x.id)) + "}"
#                     if eps_tgts else "∅")
#         lines.append(f"| {fmt_q(s)} | {role} | " + " | ".join(sym_cols) + f" | {eps_cell} |")
#     return "\n".join(lines)


# def nfa_summary(nfa, alphabet):
#     states = sorted(nfa.all_states(), key=lambda s: s.id)
#     return (f"NFA: {len(states)} states "
#             f"({{{', '.join(fmt_q(s) for s in states)}}}), "
#             f"start = {fmt_q(nfa.start)}, accept = {fmt_q(nfa.accept)}")


# # ===================
# #  7.  CoT TRACE BUILDER
# # ===================

# def _describe_structure(node, depth=0):
#     pad = "  " * depth
#     if isinstance(node, Literal):   return f"{pad}Literal '{node.char}'"
#     if isinstance(node, Epsilon):   return f"{pad}ε (epsilon)"
#     if isinstance(node, Star):      return f"{pad}Kleene Star (*)\n{_describe_structure(node.child, depth+1)}"
#     if isinstance(node, Plus):      return f"{pad}One-or-more (+)\n{_describe_structure(node.child, depth+1)}"
#     if isinstance(node, Optional_): return f"{pad}Optional (?)\n{_describe_structure(node.child, depth+1)}"
#     if isinstance(node, Concat):
#         return f"{pad}Concatenation\n{_describe_structure(node.left,depth+1)}\n{_describe_structure(node.right,depth+1)}"
#     if isinstance(node, Union):
#         return f"{pad}Union (|)\n{_describe_structure(node.left,depth+1)}\n{_describe_structure(node.right,depth+1)}"
#     return f"{pad}?"


# def _collect_rules(node, seen=None):
#     if seen is None: seen = set()
#     rules = []
#     key = type(node).__name__
#     if key not in seen:
#         seen.add(key)
#         if isinstance(node, Literal):
#             rules.append(f"**Symbol rule** for '{node.char}': two states s,a with s--'{node.char}'-->a")
#         if isinstance(node, Epsilon):
#             rules.append("**Epsilon rule**: two states s,a with s--ε-->a")
#         if isinstance(node, Concat):
#             rules.append("**Concatenation rule**: build N(r1) and N(r2), add ε from N(r1).accept to N(r2).start")
#         if isinstance(node, Union):
#             rules.append("**Union rule**: new start q0 with ε to N(r1).start and N(r2).start; both accepts ε to new accept qa")
#         if isinstance(node, Star):
#             rules.append("**Kleene Star rule**: new start q0 and accept qa; q0--ε-->N(r).start; q0--ε-->qa; N(r).accept--ε-->N(r).start; N(r).accept--ε-->qa")
#     children = ([node.left, node.right] if isinstance(node, (Concat, Union))
#                 else [node.child] if isinstance(node, (Star, Plus, Optional_)) else [])
#     for c in children:
#         rules += _collect_rules(c, seen)
#     return rules


# def _explain_eps_closure(start, lines):
#     visited, queue, steps = {start}, deque([start]), []
#     while queue:
#         s = queue.popleft()
#         for t in s.epsilon_transitions:
#             if t not in visited:
#                 visited.add(t); queue.append(t)
#                 steps.append(f"  {fmt_q(s)} --ε--> {fmt_q(t)}")
#     if steps:
#         lines.append("ε-closure BFS expansion:")
#         lines.extend(steps)
#     else:
#         lines.append("  (Start state has no outgoing ε-transitions; ε-closure = {start state only})")


# def build_cot_response(regex_str, ast_raw, nfa, alphabet):
#     lines = []
#     sa = sorted(alphabet)
#     nfa_states = sorted(nfa.all_states(), key=lambda s: s.id)

#     lines += [
#         "## Step 1: Analyse the Regular Expression\n",
#         f"Given regex: **{regex_str}**",
#         f"Alphabet Σ = {{{', '.join(sa)}}}\n",
#         "Parse tree structure:",
#         _describe_structure(ast_raw),
#     ]
#     lines += ["", "## Step 2: Thompson's Construction Rules Applied\n"]
#     for r in _collect_rules(ast_raw):
#         lines.append(f"  - {r}")

#     lines += [
#         "", "## Step 3: NFA States and Transitions\n",
#         f"NFA states: {{{', '.join(fmt_q(s) for s in nfa_states)}}}",
#         f"Start state : {fmt_q(nfa.start)}",
#         f"Accept state: {fmt_q(nfa.accept)}\n",
#         "Detailed transition listing:\n",
#     ]
#     for s in nfa_states:
#         any_out = False
#         for char, targets in sorted(s.transitions.items()):
#             tstr = "{" + ", ".join(fmt_q(t) for t in sorted(targets, key=lambda x: x.id)) + "}"
#             lines.append(f"  {fmt_q(s)} --'{char}'--> {tstr}"); any_out = True
#         if s.epsilon_transitions:
#             tstr = "{" + ", ".join(fmt_q(t) for t in sorted(s.epsilon_transitions, key=lambda x: x.id)) + "}"
#             lines.append(f"  {fmt_q(s)} --ε--> {tstr}"); any_out = True
#         if not any_out:
#             lines.append(f"  {fmt_q(s)}: (no outgoing transitions — accept sink)")

#     start_ec = epsilon_closure(frozenset([nfa.start]))
#     lines += [
#         "", "## Step 4: ε-closure of Start State\n",
#         f"ε-closure({fmt_q(nfa.start)}) = {fmt_nfa_set(start_ec)}",
#     ]
#     _explain_eps_closure(nfa.start, lines)
#     lines.append("\nThis set forms the initial configuration when processing any input string.\n")

#     lines += [
#         "", "## Step 5: NFA Transition Table (Thompson's Construction Output)\n",
#         "Each row shows all transitions for one state. '∅' = no transition.\n",
#         build_nfa_table(nfa, alphabet), "",
#         nfa_summary(nfa, alphabet),
#     ]
#     return "\n".join(lines)


# def build_nocot_response(nfa, alphabet):
#     lines = [
#         "## NFA Transition Table (Thompson's Construction)\n",
#         "The following table shows the NFA produced by Thompson's construction. "
#         "'∅' denotes no transition on that input.\n",
#         build_nfa_table(nfa, alphabet), "",
#         nfa_summary(nfa, alphabet),
#     ]
#     return "\n".join(lines)


# # ===================
# #  8.  QUESTION TEMPLATES
# # ===================

# SYSTEM_PROMPT = (
#     "You are an expert in formal language theory and automata. "
#     "When given a regular expression, you construct a Non-deterministic "
#     "Finite Automaton (NFA) using Thompson's construction. "
#     "You present the NFA as a clear transition table showing all states, "
#     "symbol transitions, and epsilon transitions."
# )

# USER_TEMPLATES = [
#     "Convert the regular expression `{regex}` over the alphabet Σ = {{{alpha}}} "
#     "to an NFA using Thompson's construction. Show the resulting NFA transition table.",
#     "Given the regex `{regex}` with alphabet {{{alpha}}}, build an NFA using "
#     "Thompson's construction and present the transition table.",
#     "Apply Thompson's NFA construction to the regular expression `{regex}` "
#     "(Σ = {{{alpha}}}). Give the NFA transition table.",
#     "Construct an NFA that accepts exactly the language described by the "
#     "regular expression `{regex}` over Σ = {{{alpha}}} using Thompson's method. "
#     "Show the transition table.",
#     (
#         "For the regular expression `{regex}` over alphabet {{{alpha}}}, "
#         "perform Thompson's NFA construction and present:\n"
#         "- All NFA states\n"
#         "- Start and accept states\n"
#         "- The full transition table (symbol transitions and ε-transitions)"
#     ),
# ]

# # OOD question templates — used for the held-out OOD eval split
# # Key differences from USER_TEMPLATES:
# #   - No mention of Thompson's construction
# #   - Different framing (informal, homework, implementation, mathematical)
# #   - No backtick regex formatting
# #   - Varied vocabulary (automaton, state machine, transition function)
# OOD_USER_TEMPLATES = [
#     "I need an NFA for the pattern {regex} using symbols {{{alpha}}}. "
#     "Please show me the full state transition table with epsilon moves.",

#     "Design a nondeterministic finite automaton that recognizes the language "
#     "defined by {regex} over {{{alpha}}}. List all states, mark which is the "
#     "start state and which are accepting, and give the complete transition table.",

#     "What NFA would you build to recognize strings matching {regex} where the "
#     "alphabet is {{{alpha}}}? Show the transition table.",

#     "For the regular expression {regex} with input alphabet {{{alpha}}}, "
#     "construct a finite automaton with nondeterminism. Present the states and "
#     "how they transition on each symbol and on epsilon.",

#     "Exercise: Given r = {regex} over alphabet {{{alpha}}}, draw (as a table) "
#     "the NFA M such that L(M) = L(r). Clearly indicate the start state and "
#     "final state.",

#     "I want to implement a regex matcher for {regex} on the alphabet {{{alpha}}}. "
#     "Build me the NFA state machine -- show every state, every transition on "
#     "input symbols, and every epsilon transition.",

#     "Suppose we want to check if strings over {{{alpha}}} match the pattern "
#     "{regex}. What NFA would do this? Give the complete transition function "
#     "as a table.",

#     "Let r = {regex} be a regular expression over Sigma = {{{alpha}}}. "
#     "Construct an NFA N = (Q, Sigma, delta, q0, F) such that L(N) = L(r). "
#     "Specify Q, q0, F, and delta in tabular form.",
# ]


# # ===================
# #  9.  DATASET ENTRY BUILDER
# # ===================

# def _build_parallel_entry(regex_str, ast_raw, alphabet, rng, tier):
#     NFAState._counter = 0
#     try:
#         ast = desugar(ast_raw)
#         nfa = thompson(ast)
#     except Exception:
#         return None

#     n_states = len(nfa.all_states())
#     cfg = TIER_CONFIG[tier]

#     # Enforce tier state bounds
#     if not (cfg["nfa_min"] <= n_states <= cfg["nfa_max"]):
#         return None

#     # Verify NFA correctness
#     try:
#         dfa = build_dfa_for_verify(nfa, alphabet)
#         if not verify(nfa, dfa, alphabet, max_len=5):
#             return None
#     except Exception:
#         return None

#     alpha_str = ", ".join(sorted(alphabet))
#     user_msg     = rng.choice(USER_TEMPLATES).format(regex=regex_str, alpha=alpha_str)
#     ood_user_msg = rng.choice(OOD_USER_TEMPLATES).format(regex=regex_str, alpha=alpha_str)

#     cot_response   = build_cot_response(regex_str, ast_raw, nfa, alphabet)
#     nocot_response = build_nocot_response(nfa, alphabet)

#     metadata = {
#         "regex":      regex_str,
#         "alphabet":   sorted(alphabet),
#         "tier":       tier,
#         "nfa_states": n_states,
#         "hash":       hashlib.md5((regex_str + str(sorted(alphabet))).encode()).hexdigest()[:8],
#     }

#     def _entry(response, question):
#         return {
#             "messages": [
#                 {"role": "system",    "content": SYSTEM_PROMPT},
#                 {"role": "user",      "content": question},
#                 {"role": "assistant", "content": response},
#             ],
#             "metadata": metadata,
#         }

#     return (
#         _entry(cot_response,   user_msg),      # cot   iid
#         _entry(nocot_response, user_msg),      # nocot iid
#         _entry(cot_response,   ood_user_msg),  # cot   ood
#         _entry(nocot_response, ood_user_msg),  # nocot ood
#     )


# # ===================
# #  10.  HANDCRAFTED EXAMPLES  (span all 5 tiers)
# # ===================

# HANDCRAFTED = [
#     # Tier 1 — simple concat/union
#     ("ab",                    ["a", "b"]),
#     ("a|b",                   ["a", "b"]),
#     ("abc",                   ["a", "b", "c"]),
#     ("ab|cd",                 ["a", "b", "c", "d"]),
#     ("abcd",                  ["a", "b", "c", "d"]),
#     ("0|1",                   ["0", "1"]),
#     ("01|10",                 ["0", "1"]),
#     # Tier 2 — closures
#     ("a*b",                   ["a", "b"]),
#     ("ab*",                   ["a", "b"]),
#     ("a+b",                   ["a", "b"]),
#     ("a?b",                   ["a", "b"]),
#     ("(ab)+",                 ["a", "b"]),
#     ("a*b*",                  ["a", "b"]),
#     ("(a|b)*",                ["a", "b"]),
#     ("(0|1)*",                ["0", "1"]),
#     ("a*(b|c)",               ["a", "b", "c"]),
#     # Tier 3 — medium
#     ("(a|b)*abb",             ["a", "b"]),
#     ("(0|1)*00",              ["0", "1"]),
#     ("(a|b)*ab",              ["a", "b"]),
#     ("(aa|b)*",               ["a", "b"]),
#     ("(ab|ba)*",              ["a", "b"]),
#     ("a*(b|c)+a*",            ["a", "b", "c"]),
#     ("(0|1)*101",             ["0", "1"]),
#     ("(a|b|c)*abc",           ["a", "b", "c"]),
#     # Tier 4 — hard
#     ("(a|b)*aba(a|b)*",       ["a", "b"]),
#     ("(0|1)*001(0|1)*",       ["0", "1"]),
#     ("(0|1)*(00|11)(0|1)*",   ["0", "1"]),
#     # Tier 5 — very hard
#     ("(a|b|c)*abc(a|b|c)*bca(a|b|c)*", ["a", "b", "c"]),
#     ("(0|1)*(010|101)(0|1)*",          ["0", "1"]),
# ]


# # ===================
# #  11.  REGEX PARSER
# # ===================

# class _Parser:
#     def __init__(self, s): self.s = s; self.pos = 0
#     def peek(self): return self.s[self.pos] if self.pos < len(self.s) else None
#     def consume(self, c=None):
#         ch = self.s[self.pos]
#         if c and ch != c: raise ValueError(f"Expected {c!r} got {ch!r}")
#         self.pos += 1; return ch
#     def parse(self):
#         node = self._union()
#         if self.pos != len(self.s): raise ValueError(f"Leftover at {self.pos}")
#         return node
#     def _union(self):
#         left = self._concat()
#         while self.peek() == '|': self.consume('|'); left = Union(left, self._concat())
#         return left
#     def _concat(self):
#         nodes = []
#         while self.peek() not in (None, ')', '|'): nodes.append(self._quantified())
#         if not nodes: raise ValueError("Empty concat")
#         result = nodes[0]
#         for n in nodes[1:]: result = Concat(result, n)
#         return result
#     def _quantified(self):
#         base = self._atom(); q = self.peek()
#         if q == '*': self.consume(); return Star(base)
#         if q == '+': self.consume(); return Plus(base)
#         if q == '?': self.consume(); return Optional_(base)
#         return base
#     def _atom(self):
#         c = self.peek()
#         if c == '(':
#             self.consume('('); node = self._union(); self.consume(')'); return node
#         if c and c not in (')', '|', '*', '+', '?'):
#             self.consume(); return Literal(c)
#         raise ValueError(f"Unexpected char {c!r}")


# def parse_regex(s): return _Parser(s).parse()


# # ===================
# #  12.  GENERATION PIPELINE
# # ===================

# def generate_handcrafted(rng):
#     pairs     = []
#     ood_pairs = []
#     for regex_str, alphabet in HANDCRAFTED:
#         try:
#             ast_raw = parse_regex(regex_str)
#         except Exception as e:
#             print(f"  [WARN] Parse failed '{regex_str}': {e}"); continue
#         result = _build_parallel_entry(regex_str, ast_raw, alphabet, rng, _guess_tier(regex_str, alphabet))
#         if result:
#             pairs.append(result[:2])   # (cot_iid, nocot_iid)
#             ood_pairs.append((result[2], result[3]))  # (cot_ood, nocot_ood)
#         else:
#             print(f"  [WARN] Build failed for '{regex_str}'")
#     return pairs, ood_pairs


# def _guess_tier(regex_str, alphabet):
#     """Rough tier guess for handcrafted examples — actual enforcement is by state count."""
#     n = len(regex_str)
#     if n <= 4:   return 1
#     if n <= 8:   return 2
#     if n <= 16:  return 3
#     if n <= 28:  return 4
#     return 5


# def generate_random_pair(rng, tier):
#     alpha_name = rng.choice(list(ALPHABETS.keys()))
#     alphabet   = ALPHABETS[alpha_name]
#     ast_raw    = random_regex(tier, alphabet, rng)
#     regex_str  = regex_to_string(ast_raw)
#     if len(regex_str) < 2: return None
#     return _build_parallel_entry(regex_str, ast_raw, alphabet, rng, tier)


# def print_stats(pairs):
#     print("\n" + "=" * 70)
#     print("  DATASET STATISTICS")
#     print("=" * 70)
#     tier_counts = {}; state_counts = []
#     for cot, _ in pairs:
#         m = cot["metadata"]
#         t = m["tier"]; tier_counts[t] = tier_counts.get(t, 0) + 1
#         state_counts.append(m["nfa_states"])
#     print(f"  Total pairs   : {len(pairs):,}")
#     for t in sorted(tier_counts):
#         cfg = TIER_CONFIG.get(t, {})
#         desc = cfg.get("desc", "")
#         bounds = f"[{cfg.get('nfa_min','?')}-{cfg.get('nfa_max','?')} states]"
#         print(f"  Tier {t} {bounds} {desc:30s}: {tier_counts[t]:,}")
#     print(f"  NFA states    : avg={sum(state_counts)/len(state_counts):.1f}, "
#           f"min={min(state_counts)}, max={max(state_counts)}")
#     print("=" * 70 + "\n")


# def write_split(data, path):
#     with open(path, "w", encoding="utf-8") as f:
#         for ex in data:
#             f.write(json.dumps(ex, ensure_ascii=False) + "\n")
#     print(f"  Wrote {len(data):,} examples -> {path}")


# # ===================
# #  13.  ENTRY POINT
# # ===================

# def main():
#     ap = argparse.ArgumentParser(description="Generate 5-tier parallel CoT/No-CoT NFA datasets")
#     ap.add_argument("--n",          type=int,   default=25000, help="Target pairs")
#     ap.add_argument("--seed",       type=int,   default=42)
#     ap.add_argument("--out_prefix", type=str,   default="nfa_v2")
#     ap.add_argument("--val_split",  type=float, default=0.05)
#     ap.add_argument("--test_split", type=float, default=0.10,
#                     help="Larger test split (10%) for more reliable evaluation")
#     ap.add_argument("--test_per_tier", type=int, default=200,
#                     help="Minimum test examples per tier (stratified)")
#     args = ap.parse_args()

#     rng   = random.Random(args.seed)
#     # Each entry is a 4-tuple: (cot_iid, nocot_iid, cot_ood, nocot_ood)
#     # Keeping them together ensures IID and OOD always share the same regex/answer
#     quads = []
#     seen  = set()

#     print("Generating handcrafted examples...")
#     hc_pairs, hc_ood = generate_handcrafted(rng)
#     for (c_iid, nc_iid), (c_ood, nc_ood) in zip(hc_pairs, hc_ood):
#         h = c_iid["metadata"]["hash"]
#         if h not in seen:
#             seen.add(h)
#             quads.append((c_iid, nc_iid, c_ood, nc_ood))
#     print(f"  Added {len(quads)} handcrafted quads.")

#     print(f"Generating up to {args.n:,} random pairs (5 tiers)...")
#     max_attempts = args.n * 40

#     with tqdm(total=args.n) as pbar:
#         pbar.update(len(quads))
#         attempts = 0
#         while len(quads) < args.n and attempts < max_attempts:
#             attempts += 1
#             tier   = rng.choices(list(TIER_DIST), weights=list(TIER_DIST.values()))[0]
#             result = generate_random_pair(rng, tier)
#             if result is None: continue
#             h = result[0]["metadata"]["hash"]
#             if h in seen: continue
#             seen.add(h)
#             quads.append((result[0], result[1], result[2], result[3]))
#             pbar.update(1)

#     print(f"\nGenerated {len(quads):,} quads in {attempts:,} attempts.")
#     print_stats([(q[0],q[1]) for q in quads])

#     # Stratified split — kept as quads so IID/OOD stay perfectly aligned
#     rng.shuffle(quads)
#     by_tier = defaultdict(list)
#     for quad in quads:
#         by_tier[quad[0]["metadata"]["tier"]].append(quad)

#     test_quads  = []
#     train_quads = []
#     for tier in sorted(by_tier):
#         tier_list = by_tier[tier]
#         n_test    = min(args.test_per_tier, len(tier_list) // 5)
#         test_quads.extend(tier_list[:n_test])
#         train_quads.extend(tier_list[n_test:])

#     # Val split from remaining train
#     rng.shuffle(train_quads)
#     n_val        = int(len(train_quads) * args.val_split)
#     val_quads    = train_quads[:n_val]
#     train_quads  = train_quads[n_val:]

#     p = args.out_prefix
#     print(f"\nSplit: train={len(train_quads):,} | val={len(val_quads):,} | test={len(test_quads):,}")
#     print(f"Test is stratified: ~{args.test_per_tier} examples per tier\n")

#     # IID splits (training templates)
#     for split_name, split_data in [("train", train_quads), ("val", val_quads), ("test", test_quads)]:
#         write_split([q[0] for q in split_data], f"{p}_cot_{split_name}.jsonl")
#         write_split([q[1] for q in split_data], f"{p}_nocot_{split_name}.jsonl")

#     write_split([q[0] for q in quads], f"{p}_cot_full.jsonl")
#     write_split([q[1] for q in quads], f"{p}_nocot_full.jsonl")

#     # OOD splits — same quads, same indices, just use q[2] and q[3]
#     # This guarantees perfect alignment: OOD[i] and IID[i] are same regex, same answer
#     write_split([q[2] for q in test_quads],  f"{p}_cot_ood_test.jsonl")
#     write_split([q[3] for q in test_quads],  f"{p}_nocot_ood_test.jsonl")
#     write_split([q[2] for q in quads],       f"{p}_cot_ood_full.jsonl")
#     write_split([q[3] for q in quads],       f"{p}_nocot_ood_full.jsonl")
#     print(f"  OOD test: {len(test_quads)} pairs (perfectly aligned with IID test)")

#     # Per-tier test breakdown
#     print("\nTest set per tier:")
#     tier_test = defaultdict(int)
#     for quad in test_quads:
#         tier_test[quad[0]["metadata"]["tier"]] += 1
#     for t in sorted(tier_test):
#         cfg = TIER_CONFIG[t]
#         print(f"  Tier {t} [{cfg['nfa_min']}-{cfg['nfa_max']} states]: {tier_test[t]} examples")

#     print("\nDone.")


# if __name__ == "__main__":
#     main()























"""
Parallel NFA Dataset Generator v2 — 5-Tier Difficulty
=======================================================
Generates TWO parallel datasets from the same regex instances:
  1. CoT   dataset : question + Thompson NFA steps + final NFA table
  2. NoCoT dataset : question + final NFA table ONLY

5 tiers by NFA state count (complexity proxy):
  Tier 1 : 4-8   states  — concat + union only             (trivial)
  Tier 2 : 8-14  states  — + star/plus/optional            (easy)
  Tier 3 : 14-22 states  — + star_union/star_concat        (medium)
  Tier 4 : 22-32 states  — + nested closures               (hard)
  Tier 5 : 32-50 states  — + deeply nested/chained ops     (very hard)

Test set is stratified: equal examples per tier so difficulty is controlled.

Output files (JSONL, Qwen ChatML):
  nfa_v2_cot_train.jsonl / nfa_v2_cot_val.jsonl / nfa_v2_cot_test.jsonl
  nfa_v2_nocot_train.jsonl / nfa_v2_nocot_val.jsonl / nfa_v2_nocot_test.jsonl
  nfa_v2_cot_full.jsonl / nfa_v2_nocot_full.jsonl
"""

import random
import json
import hashlib
import itertools
import argparse
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, **kwargs): self.n = 0; self.total = total
        def update(self, n=1):
            self.n += n
            print(f"\r  {self.n}/{self.total}", end="", flush=True)
        def __enter__(self): return self
        def __exit__(self, *a): print()


# ===================
#  1.  REGEX AST
# ===================

class RegexNode: pass
class Literal(RegexNode):
    def __init__(self, char): self.char = char
class Epsilon(RegexNode): pass
class Concat(RegexNode):
    def __init__(self, l, r): self.left = l; self.right = r
class Union(RegexNode):
    def __init__(self, l, r): self.left = l; self.right = r
class Star(RegexNode):
    def __init__(self, c): self.child = c
class Plus(RegexNode):
    def __init__(self, c): self.child = c
class Optional_(RegexNode):
    def __init__(self, c): self.child = c


def desugar(node):
    if isinstance(node, (Literal, Epsilon)): return node
    if isinstance(node, Concat):   return Concat(desugar(node.left), desugar(node.right))
    if isinstance(node, Union):    return Union(desugar(node.left), desugar(node.right))
    if isinstance(node, Star):     return Star(desugar(node.child))
    if isinstance(node, Plus):
        d = desugar(node.child); return Concat(d, Star(deepcopy(d)))
    if isinstance(node, Optional_):return Union(desugar(node.child), Epsilon())
    raise ValueError(f"Unknown node {node}")


def regex_to_string(node, parent_prec=0):
    if isinstance(node, Literal):   return node.char
    if isinstance(node, Epsilon):   return "eps"
    if isinstance(node, Star):
        inner = regex_to_string(node.child, 3)
        return f"({inner})*" if isinstance(node.child, (Concat, Union)) else f"{inner}*"
    if isinstance(node, Plus):
        inner = regex_to_string(node.child, 3)
        return f"({inner})+" if isinstance(node.child, (Concat, Union)) else f"{inner}+"
    if isinstance(node, Optional_):
        inner = regex_to_string(node.child, 3)
        return f"({inner})?" if isinstance(node.child, (Concat, Union)) else f"{inner}?"
    if isinstance(node, Concat):
        l = regex_to_string(node.left, 2); r = regex_to_string(node.right, 2)
        s = l + r; return f"({s})" if parent_prec > 2 else s
    if isinstance(node, Union):
        l = regex_to_string(node.left, 1); r = regex_to_string(node.right, 1)
        s = f"{l}|{r}"; return f"({s})" if parent_prec > 1 else s
    return "?"


# ===================
#  2.  ALPHABETS
# ===================

ALPHABETS = {
    "binary": ["0", "1"],
    "ab":     ["a", "b"],
    "abc":    ["a", "b", "c"],
    "abcd":   ["a", "b", "c", "d"],
    "digits": ["0", "1", "2", "3"],
}

# ===================
#  3.  5-TIER RANDOM REGEX GENERATOR
# ===================

# Per-tier configuration
TIER_CONFIG = {
    1: {
        "max_depth": 3,
        "base_leaf": 0.55,
        "ops": ["concat", "union", "concat3"],
        "weights": [0.55, 0.35, 0.10],
        "nfa_min": 4, "nfa_max": 8,
        "desc": "concat+union only",
    },
    2: {
        "max_depth": 4,
        "base_leaf": 0.42,
        "ops": ["concat", "union", "star", "plus", "optional", "concat3"],
        "weights": [0.28, 0.20, 0.22, 0.14, 0.08, 0.08],
        "nfa_min": 8, "nfa_max": 14,
        "desc": "closures introduced",
    },
    3: {
        "max_depth": 5,
        "base_leaf": 0.30,
        "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat"],
        "weights": [0.22, 0.16, 0.20, 0.12, 0.10, 0.12, 0.08],
        "nfa_min": 14, "nfa_max": 22,
        "desc": "star_union/star_concat",
    },
    4: {
        "max_depth": 6,
        "base_leaf": 0.22,
        "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat", "concat4"],
        "weights": [0.18, 0.14, 0.20, 0.10, 0.10, 0.12, 0.08, 0.08],
        "nfa_min": 22, "nfa_max": 32,
        "desc": "nested closures + concat4",
    },
    5: {
        "max_depth": 8,
        "base_leaf": 0.15,
        "ops": ["concat", "union", "star", "plus", "concat3", "star_union", "star_concat",
                "concat4", "star_star", "union_chain"],
        "weights": [0.14, 0.10, 0.18, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08, 0.06],
        "nfa_min": 32, "nfa_max": 50,
        "desc": "deeply nested/chained",
    },
}

# Tier distribution for training set
TIER_DIST = {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.20}


def random_regex(tier: int, alphabet: list, rng: random.Random, depth: int = 0) -> RegexNode:
    cfg       = TIER_CONFIG[tier]
    max_depth = cfg["max_depth"]
    base_leaf = cfg["base_leaf"]
    leaf_prob = min(0.95, base_leaf + 0.13 * depth)

    if depth >= max_depth or rng.random() < leaf_prob:
        return Literal(rng.choice(alphabet))

    op = rng.choices(cfg["ops"], weights=cfg["weights"])[0]

    def sub(d=1): return random_regex(tier, alphabet, rng, depth + d)

    if op == "concat":      return Concat(sub(), sub())
    if op == "concat3":     return Concat(Concat(sub(), sub()), sub())
    if op == "concat4":     return Concat(Concat(Concat(sub(), sub()), sub()), sub())
    if op == "union":       return Union(sub(), sub())
    if op == "union_chain": return Union(Union(sub(), sub()), sub())
    if op == "star":        return Star(sub())
    if op == "plus":        return Plus(sub())
    if op == "optional":    return Optional_(sub())
    if op == "star_union":  return Star(Union(sub(), sub()))
    if op == "star_concat": return Star(Concat(sub(), sub()))
    if op == "star_star":   return Star(Star(sub()))  # (r*)* = r* but produces complex NFA
    return Literal(rng.choice(alphabet))


# ===================
#  4.  THOMPSON'S NFA
# ===================

class NFAState:
    _counter = 0
    def __init__(self):
        NFAState._counter += 1
        self.id = NFAState._counter
        self.transitions  = defaultdict(list)
        self.epsilon_transitions = []
    def __repr__(self):  return f"q{self.id}"
    def __hash__(self):  return hash(self.id)
    def __eq__(self, o): return isinstance(o, NFAState) and self.id == o.id


class NFA:
    def __init__(self, start, accept):
        self.start  = start
        self.accept = accept

    def all_states(self):
        visited, stack = set(), [self.start]
        while stack:
            s = stack.pop()
            if s in visited: continue
            visited.add(s)
            for tgts in s.transitions.values():
                stack.extend(t for t in tgts if t not in visited)
            stack.extend(t for t in s.epsilon_transitions if t not in visited)
        return visited


def thompson(node) -> NFA:
    if isinstance(node, Literal):
        s, a = NFAState(), NFAState()
        s.transitions[node.char].append(a)
        return NFA(s, a)
    if isinstance(node, Epsilon):
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.append(a)
        return NFA(s, a)
    if isinstance(node, Concat):
        n1 = thompson(node.left); n2 = thompson(node.right)
        n1.accept.epsilon_transitions.append(n2.start)
        return NFA(n1.start, n2.accept)
    if isinstance(node, Union):
        n1 = thompson(node.left); n2 = thompson(node.right)
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.extend([n1.start, n2.start])
        n1.accept.epsilon_transitions.append(a)
        n2.accept.epsilon_transitions.append(a)
        return NFA(s, a)
    if isinstance(node, Star):
        n = thompson(node.child)
        s, a = NFAState(), NFAState()
        s.epsilon_transitions.extend([n.start, a])
        n.accept.epsilon_transitions.extend([n.start, a])
        return NFA(s, a)
    raise ValueError(f"thompson: unexpected {type(node)}")


# ===================
#  5.  VERIFICATION
# ===================

def epsilon_closure(states):
    stack, closure = list(states), set(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon_transitions:
            if t not in closure:
                closure.add(t); stack.append(t)
    return frozenset(closure)


def nfa_move(states, char):
    result = set()
    for s in states:
        result.update(s.transitions.get(char, []))
    return frozenset(result)


def nfa_accepts(nfa, s):
    states = epsilon_closure(frozenset([nfa.start]))
    for c in s:
        states = epsilon_closure(nfa_move(states, c))
    return nfa.accept in states


def build_dfa_for_verify(nfa, alphabet):
    """Minimal DFA build just for cross-checking — not exposed in dataset."""
    from collections import deque as dq
    class DFA:
        def __init__(self): self.transitions={}; self.start=0; self.accept_states=set(); self.states=[]
        def accepts(self, s):
            cur = self.start
            for c in s:
                if (cur,c) not in self.transitions: return False
                cur = self.transitions[(cur,c)]
            return cur in self.accept_states
    dfa = DFA()
    sc = epsilon_closure(frozenset([nfa.start]))
    dfa.states.append(sc); idx={sc:0}
    if nfa.accept in sc: dfa.accept_states.add(0)
    q = dq([sc])
    while q:
        cur = q.popleft(); ci = idx[cur]
        for c in sorted(alphabet):
            nxt = epsilon_closure(nfa_move(cur, c))
            if not nxt: continue
            if nxt not in idx:
                i = len(dfa.states); dfa.states.append(nxt); idx[nxt]=i
                if nfa.accept in nxt: dfa.accept_states.add(i)
                q.append(nxt)
            dfa.transitions[(ci,c)] = idx[nxt]
    return dfa


def verify(nfa, dfa, alphabet, max_len=5):
    for length in range(max_len+1):
        for combo in itertools.product(alphabet, repeat=length):
            s = "".join(combo)
            if nfa_accepts(nfa, s) != dfa.accepts(s):
                return False
    return True


# ===================
#  6.  NFA TABLE FORMATTER
# ===================

def fmt_q(s): return f"q{s.id}"

def fmt_nfa_set(fs):
    ids = sorted(s.id for s in fs)
    return "{" + ", ".join(f"q{i}" for i in ids) + "}"


def build_nfa_table(nfa, alphabet):
    sa = sorted(alphabet)
    states = sorted(nfa.all_states(), key=lambda s: s.id)
    header = "| State | Role | " + " | ".join(sa) + " | ε (epsilon) |"
    sep    = "|-------|------|" + "|".join(["--------"]*len(sa)) + "|-------------|"
    lines  = [header, sep]
    for s in states:
        role = ""
        if s == nfa.start and s == nfa.accept: role = "start, accept"
        elif s == nfa.start:  role = "start"
        elif s == nfa.accept: role = "accept"
        sym_cols = []
        for c in sa:
            tgts = s.transitions.get(c, [])
            sym_cols.append("{" + ", ".join(fmt_q(t) for t in sorted(tgts, key=lambda x: x.id)) + "}"
                            if tgts else "∅")
        eps_tgts = s.epsilon_transitions
        eps_cell = ("{" + ", ".join(fmt_q(t) for t in sorted(eps_tgts, key=lambda x: x.id)) + "}"
                    if eps_tgts else "∅")
        lines.append(f"| {fmt_q(s)} | {role} | " + " | ".join(sym_cols) + f" | {eps_cell} |")
    return "\n".join(lines)


def nfa_summary(nfa, alphabet):
    states = sorted(nfa.all_states(), key=lambda s: s.id)
    return (f"NFA: {len(states)} states "
            f"({{{', '.join(fmt_q(s) for s in states)}}}), "
            f"start = {fmt_q(nfa.start)}, accept = {fmt_q(nfa.accept)}")


# ===================
#  7.  CoT TRACE BUILDER
# ===================

def _describe_structure(node, depth=0):
    pad = "  " * depth
    if isinstance(node, Literal):   return f"{pad}Literal '{node.char}'"
    if isinstance(node, Epsilon):   return f"{pad}ε (epsilon)"
    if isinstance(node, Star):      return f"{pad}Kleene Star (*)\n{_describe_structure(node.child, depth+1)}"
    if isinstance(node, Plus):      return f"{pad}One-or-more (+)\n{_describe_structure(node.child, depth+1)}"
    if isinstance(node, Optional_): return f"{pad}Optional (?)\n{_describe_structure(node.child, depth+1)}"
    if isinstance(node, Concat):
        return f"{pad}Concatenation\n{_describe_structure(node.left,depth+1)}\n{_describe_structure(node.right,depth+1)}"
    if isinstance(node, Union):
        return f"{pad}Union (|)\n{_describe_structure(node.left,depth+1)}\n{_describe_structure(node.right,depth+1)}"
    return f"{pad}?"


def _collect_rules(node, seen=None):
    if seen is None: seen = set()
    rules = []
    key = type(node).__name__
    if key not in seen:
        seen.add(key)
        if isinstance(node, Literal):
            rules.append(f"**Symbol rule** for '{node.char}': two states s,a with s--'{node.char}'-->a")
        if isinstance(node, Epsilon):
            rules.append("**Epsilon rule**: two states s,a with s--ε-->a")
        if isinstance(node, Concat):
            rules.append("**Concatenation rule**: build N(r1) and N(r2), add ε from N(r1).accept to N(r2).start")
        if isinstance(node, Union):
            rules.append("**Union rule**: new start q0 with ε to N(r1).start and N(r2).start; both accepts ε to new accept qa")
        if isinstance(node, Star):
            rules.append("**Kleene Star rule**: new start q0 and accept qa; q0--ε-->N(r).start; q0--ε-->qa; N(r).accept--ε-->N(r).start; N(r).accept--ε-->qa")
    children = ([node.left, node.right] if isinstance(node, (Concat, Union))
                else [node.child] if isinstance(node, (Star, Plus, Optional_)) else [])
    for c in children:
        rules += _collect_rules(c, seen)
    return rules


def _explain_eps_closure(start, lines):
    visited, queue, steps = {start}, deque([start]), []
    while queue:
        s = queue.popleft()
        for t in s.epsilon_transitions:
            if t not in visited:
                visited.add(t); queue.append(t)
                steps.append(f"  {fmt_q(s)} --ε--> {fmt_q(t)}")
    if steps:
        lines.append("ε-closure BFS expansion:")
        lines.extend(steps)
    else:
        lines.append("  (Start state has no outgoing ε-transitions; ε-closure = {start state only})")


def build_cot_response(regex_str, ast_raw, nfa, alphabet):
    lines = []
    sa = sorted(alphabet)
    nfa_states = sorted(nfa.all_states(), key=lambda s: s.id)

    lines += [
        "## Step 1: Analyse the Regular Expression\n",
        f"Given regex: **{regex_str}**",
        f"Alphabet Σ = {{{', '.join(sa)}}}\n",
        "Parse tree structure:",
        _describe_structure(ast_raw),
    ]
    lines += ["", "## Step 2: Thompson's Construction Rules Applied\n"]
    for r in _collect_rules(ast_raw):
        lines.append(f"  - {r}")

    lines += [
        "", "## Step 3: NFA States and Transitions\n",
        f"NFA states: {{{', '.join(fmt_q(s) for s in nfa_states)}}}",
        f"Start state : {fmt_q(nfa.start)}",
        f"Accept state: {fmt_q(nfa.accept)}\n",
        "Detailed transition listing:\n",
    ]
    for s in nfa_states:
        any_out = False
        for char, targets in sorted(s.transitions.items()):
            tstr = "{" + ", ".join(fmt_q(t) for t in sorted(targets, key=lambda x: x.id)) + "}"
            lines.append(f"  {fmt_q(s)} --'{char}'--> {tstr}"); any_out = True
        if s.epsilon_transitions:
            tstr = "{" + ", ".join(fmt_q(t) for t in sorted(s.epsilon_transitions, key=lambda x: x.id)) + "}"
            lines.append(f"  {fmt_q(s)} --ε--> {tstr}"); any_out = True
        if not any_out:
            lines.append(f"  {fmt_q(s)}: (no outgoing transitions — accept sink)")

    start_ec = epsilon_closure(frozenset([nfa.start]))
    lines += [
        "", "## Step 4: ε-closure of Start State\n",
        f"ε-closure({fmt_q(nfa.start)}) = {fmt_nfa_set(start_ec)}",
    ]
    _explain_eps_closure(nfa.start, lines)
    lines.append("\nThis set forms the initial configuration when processing any input string.\n")

    lines += [
        "", "## Step 5: NFA Transition Table (Thompson's Construction Output)\n",
        "Each row shows all transitions for one state. '∅' = no transition.\n",
        build_nfa_table(nfa, alphabet), "",
        nfa_summary(nfa, alphabet),
    ]
    return "\n".join(lines)


def build_nocot_response(nfa, alphabet):
    lines = [
        "## NFA Transition Table (Thompson's Construction)\n",
        "The following table shows the NFA produced by Thompson's construction. "
        "'∅' denotes no transition on that input.\n",
        build_nfa_table(nfa, alphabet), "",
        nfa_summary(nfa, alphabet),
    ]
    return "\n".join(lines)


# ===================
#  8.  QUESTION TEMPLATES
# ===================

SYSTEM_PROMPT = (
    "You are an expert in formal language theory and automata. "
    "When given a regular expression, you construct a Non-deterministic "
    "Finite Automaton (NFA) using Thompson's construction. "
    "You present the NFA as a clear transition table showing all states, "
    "symbol transitions, and epsilon transitions."
)

USER_TEMPLATES = [
    "Convert the regular expression `{regex}` over the alphabet Σ = {{{alpha}}} "
    "to an NFA using Thompson's construction. Show the resulting NFA transition table.",
    "Given the regex `{regex}` with alphabet {{{alpha}}}, build an NFA using "
    "Thompson's construction and present the transition table.",
    "Apply Thompson's NFA construction to the regular expression `{regex}` "
    "(Σ = {{{alpha}}}). Give the NFA transition table.",
    "Construct an NFA that accepts exactly the language described by the "
    "regular expression `{regex}` over Σ = {{{alpha}}} using Thompson's method. "
    "Show the transition table.",
    (
        "For the regular expression `{regex}` over alphabet {{{alpha}}}, "
        "perform Thompson's NFA construction and present:\n"
        "- All NFA states\n"
        "- Start and accept states\n"
        "- The full transition table (symbol transitions and ε-transitions)"
    ),
]

# OOD question templates — used for the held-out OOD eval split
# Key differences from USER_TEMPLATES:
#   - No mention of Thompson's construction
#   - Different framing (informal, homework, implementation, mathematical)
#   - No backtick regex formatting
#   - Varied vocabulary (automaton, state machine, transition function)
OOD_USER_TEMPLATES = [
    "I need an NFA for the pattern {regex} using symbols {{{alpha}}}. "
    "Please show me the full state transition table with epsilon moves.",

    "Design a nondeterministic finite automaton that recognizes the language "
    "defined by {regex} over {{{alpha}}}. List all states, mark which is the "
    "start state and which are accepting, and give the complete transition table.",

    "What NFA would you build to recognize strings matching {regex} where the "
    "alphabet is {{{alpha}}}? Show the transition table.",

    "For the regular expression {regex} with input alphabet {{{alpha}}}, "
    "construct a finite automaton with nondeterminism. Present the states and "
    "how they transition on each symbol and on epsilon.",

    "Exercise: Given r = {regex} over alphabet {{{alpha}}}, draw (as a table) "
    "the NFA M such that L(M) = L(r). Clearly indicate the start state and "
    "final state.",

    "I want to implement a regex matcher for {regex} on the alphabet {{{alpha}}}. "
    "Build me the NFA state machine -- show every state, every transition on "
    "input symbols, and every epsilon transition.",

    "Suppose we want to check if strings over {{{alpha}}} match the pattern "
    "{regex}. What NFA would do this? Give the complete transition function "
    "as a table.",

    "Let r = {regex} be a regular expression over Sigma = {{{alpha}}}. "
    "Construct an NFA N = (Q, Sigma, delta, q0, F) such that L(N) = L(r). "
    "Specify Q, q0, F, and delta in tabular form.",
]


# ===================
#  9.  DATASET ENTRY BUILDER
# ===================

def _build_parallel_entry(regex_str, ast_raw, alphabet, rng, tier):
    NFAState._counter = 0
    try:
        ast = desugar(ast_raw)
        nfa = thompson(ast)
    except Exception:
        return None

    n_states = len(nfa.all_states())
    cfg = TIER_CONFIG[tier]

    # Enforce tier state bounds
    if not (cfg["nfa_min"] <= n_states <= cfg["nfa_max"]):
        return None

    # Verify NFA correctness
    try:
        dfa = build_dfa_for_verify(nfa, alphabet)
        if not verify(nfa, dfa, alphabet, max_len=5):
            return None
    except Exception:
        return None

    alpha_str = ", ".join(sorted(alphabet))
    user_msg     = rng.choice(USER_TEMPLATES).format(regex=regex_str, alpha=alpha_str)
    ood_user_msg = rng.choice(OOD_USER_TEMPLATES).format(regex=regex_str, alpha=alpha_str)

    cot_response   = build_cot_response(regex_str, ast_raw, nfa, alphabet)
    nocot_response = build_nocot_response(nfa, alphabet)

    metadata = {
        "regex":      regex_str,
        "alphabet":   sorted(alphabet),
        "tier":       tier,
        "nfa_states": n_states,
        "hash":       hashlib.md5((regex_str + str(sorted(alphabet))).encode()).hexdigest()[:8],
    }

    def _entry(response, question):
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response},
            ],
            "metadata": metadata,
        }

    return (
        _entry(cot_response,   user_msg),      # cot   iid
        _entry(nocot_response, user_msg),      # nocot iid
        _entry(cot_response,   ood_user_msg),  # cot   ood
        _entry(nocot_response, ood_user_msg),  # nocot ood
    )


# ===================
#  10.  HANDCRAFTED EXAMPLES  (span all 5 tiers)
# ===================

HANDCRAFTED = [
    # Tier 1 — simple concat/union
    ("ab",                    ["a", "b"]),
    ("a|b",                   ["a", "b"]),
    ("abc",                   ["a", "b", "c"]),
    ("ab|cd",                 ["a", "b", "c", "d"]),
    ("abcd",                  ["a", "b", "c", "d"]),
    ("0|1",                   ["0", "1"]),
    ("01|10",                 ["0", "1"]),
    # Tier 2 — closures
    ("a*b",                   ["a", "b"]),
    ("ab*",                   ["a", "b"]),
    ("a+b",                   ["a", "b"]),
    ("a?b",                   ["a", "b"]),
    ("(ab)+",                 ["a", "b"]),
    ("a*b*",                  ["a", "b"]),
    ("(a|b)*",                ["a", "b"]),
    ("(0|1)*",                ["0", "1"]),
    ("a*(b|c)",               ["a", "b", "c"]),
    # Tier 3 — medium
    ("(a|b)*abb",             ["a", "b"]),
    ("(0|1)*00",              ["0", "1"]),
    ("(a|b)*ab",              ["a", "b"]),
    ("(aa|b)*",               ["a", "b"]),
    ("(ab|ba)*",              ["a", "b"]),
    ("a*(b|c)+a*",            ["a", "b", "c"]),
    ("(0|1)*101",             ["0", "1"]),
    ("(a|b|c)*abc",           ["a", "b", "c"]),
    # Tier 4 — hard
    ("(a|b)*aba(a|b)*",       ["a", "b"]),
    ("(0|1)*001(0|1)*",       ["0", "1"]),
    ("(0|1)*(00|11)(0|1)*",   ["0", "1"]),
    # Tier 5 — very hard
    ("(a|b|c)*abc(a|b|c)*bca(a|b|c)*", ["a", "b", "c"]),
    ("(0|1)*(010|101)(0|1)*",          ["0", "1"]),
]


# ===================
#  11.  REGEX PARSER
# ===================

class _Parser:
    def __init__(self, s): self.s = s; self.pos = 0
    def peek(self): return self.s[self.pos] if self.pos < len(self.s) else None
    def consume(self, c=None):
        ch = self.s[self.pos]
        if c and ch != c: raise ValueError(f"Expected {c!r} got {ch!r}")
        self.pos += 1; return ch
    def parse(self):
        node = self._union()
        if self.pos != len(self.s): raise ValueError(f"Leftover at {self.pos}")
        return node
    def _union(self):
        left = self._concat()
        while self.peek() == '|': self.consume('|'); left = Union(left, self._concat())
        return left
    def _concat(self):
        nodes = []
        while self.peek() not in (None, ')', '|'): nodes.append(self._quantified())
        if not nodes: raise ValueError("Empty concat")
        result = nodes[0]
        for n in nodes[1:]: result = Concat(result, n)
        return result
    def _quantified(self):
        base = self._atom(); q = self.peek()
        if q == '*': self.consume(); return Star(base)
        if q == '+': self.consume(); return Plus(base)
        if q == '?': self.consume(); return Optional_(base)
        return base
    def _atom(self):
        c = self.peek()
        if c == '(':
            self.consume('('); node = self._union(); self.consume(')'); return node
        if c and c not in (')', '|', '*', '+', '?'):
            self.consume(); return Literal(c)
        raise ValueError(f"Unexpected char {c!r}")


def parse_regex(s): return _Parser(s).parse()


# ===================
#  12.  GENERATION PIPELINE
# ===================

def generate_handcrafted(rng):
    pairs     = []
    ood_pairs = []
    for regex_str, alphabet in HANDCRAFTED:
        try:
            ast_raw = parse_regex(regex_str)
        except Exception as e:
            print(f"  [WARN] Parse failed '{regex_str}': {e}"); continue
        result = _build_parallel_entry(regex_str, ast_raw, alphabet, rng, _guess_tier(regex_str, alphabet))
        if result:
            pairs.append(result[:2])   # (cot_iid, nocot_iid)
            ood_pairs.append((result[2], result[3]))  # (cot_ood, nocot_ood)
        else:
            print(f"  [WARN] Build failed for '{regex_str}'")
    return pairs, ood_pairs


def _guess_tier(regex_str, alphabet):
    """Rough tier guess for handcrafted examples — actual enforcement is by state count."""
    n = len(regex_str)
    if n <= 4:   return 1
    if n <= 8:   return 2
    if n <= 16:  return 3
    if n <= 28:  return 4
    return 5


def generate_random_pair(rng, tier):
    alpha_name = rng.choice(list(ALPHABETS.keys()))
    alphabet   = ALPHABETS[alpha_name]
    ast_raw    = random_regex(tier, alphabet, rng)
    regex_str  = regex_to_string(ast_raw)
    if len(regex_str) < 2: return None
    return _build_parallel_entry(regex_str, ast_raw, alphabet, rng, tier)


def print_stats(pairs):
    print("\n" + "=" * 70)
    print("  DATASET STATISTICS")
    print("=" * 70)
    tier_counts = {}; state_counts = []
    for cot, _ in pairs:
        m = cot["metadata"]
        t = m["tier"]; tier_counts[t] = tier_counts.get(t, 0) + 1
        state_counts.append(m["nfa_states"])
    print(f"  Total pairs   : {len(pairs):,}")
    for t in sorted(tier_counts):
        cfg = TIER_CONFIG.get(t, {})
        desc = cfg.get("desc", "")
        bounds = f"[{cfg.get('nfa_min','?')}-{cfg.get('nfa_max','?')} states]"
        print(f"  Tier {t} {bounds} {desc:30s}: {tier_counts[t]:,}")
    print(f"  NFA states    : avg={sum(state_counts)/len(state_counts):.1f}, "
          f"min={min(state_counts)}, max={max(state_counts)}")
    print("=" * 70 + "\n")


def write_split(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data):,} examples -> {path}")


# ===================
#  13.  ENTRY POINT
# ===================

def main():
    ap = argparse.ArgumentParser(description="Generate 5-tier parallel CoT/No-CoT NFA datasets")
    ap.add_argument("--n",          type=int,   default=25000, help="Target pairs")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out_prefix", type=str,   default="nfa_v2")
    ap.add_argument("--val_split",  type=float, default=0.05)
    ap.add_argument("--test_split", type=float, default=0.10,
                    help="Larger test split (10%) for more reliable evaluation")
    ap.add_argument("--test_per_tier", type=int, default=200,
                    help="Minimum test examples per tier (stratified)")
    args = ap.parse_args()

    rng   = random.Random(args.seed)
    # Each entry is a 4-tuple: (cot_iid, nocot_iid, cot_ood, nocot_ood)
    # Keeping them together ensures IID and OOD always share the same regex/answer
    quads = []
    seen  = set()

    print("Generating handcrafted examples...")
    hc_pairs, hc_ood = generate_handcrafted(rng)
    for (c_iid, nc_iid), (c_ood, nc_ood) in zip(hc_pairs, hc_ood):
        h = c_iid["metadata"]["hash"]
        if h not in seen:
            seen.add(h)
            quads.append((c_iid, nc_iid, c_ood, nc_ood))
    print(f"  Added {len(quads)} handcrafted quads.")

    print(f"Generating up to {args.n:,} random pairs (5 tiers)...")
    max_attempts = args.n * 40

    with tqdm(total=args.n) as pbar:
        pbar.update(len(quads))
        attempts = 0
        while len(quads) < args.n and attempts < max_attempts:
            attempts += 1
            tier   = rng.choices(list(TIER_DIST), weights=list(TIER_DIST.values()))[0]
            result = generate_random_pair(rng, tier)
            if result is None: continue
            h = result[0]["metadata"]["hash"]
            if h in seen: continue
            seen.add(h)
            quads.append((result[0], result[1], result[2], result[3]))
            pbar.update(1)

    print(f"\nGenerated {len(quads):,} quads in {attempts:,} attempts.")
    print_stats([(q[0],q[1]) for q in quads])

    # Stratified split — kept as quads so IID/OOD stay perfectly aligned
    rng.shuffle(quads)
    by_tier = defaultdict(list)
    for quad in quads:
        by_tier[quad[0]["metadata"]["tier"]].append(quad)

    test_quads  = []
    train_quads = []
    for tier in sorted(by_tier):
        tier_list = by_tier[tier]
        n_test    = min(args.test_per_tier, len(tier_list) // 5)
        test_quads.extend(tier_list[:n_test])
        train_quads.extend(tier_list[n_test:])

    # Val split from remaining train
    rng.shuffle(train_quads)
    n_val        = int(len(train_quads) * args.val_split)
    val_quads    = train_quads[:n_val]
    train_quads  = train_quads[n_val:]

    p = args.out_prefix
    print(f"\nSplit: train={len(train_quads):,} | val={len(val_quads):,} | test={len(test_quads):,}")
    print(f"Test is stratified: ~{args.test_per_tier} examples per tier\n")

    # IID splits (training templates)
    for split_name, split_data in [("train", train_quads), ("val", val_quads), ("test", test_quads)]:
        write_split([q[0] for q in split_data], f"{p}_cot_{split_name}.jsonl")
        write_split([q[1] for q in split_data], f"{p}_nocot_{split_name}.jsonl")

    write_split([q[0] for q in quads], f"{p}_cot_full.jsonl")
    write_split([q[1] for q in quads], f"{p}_nocot_full.jsonl")

    # OOD splits — same quads, same indices, just use q[2] and q[3]
    # This guarantees perfect alignment: OOD[i] and IID[i] are same regex, same answer
    write_split([q[2] for q in test_quads],  f"{p}_cot_ood_test.jsonl")
    write_split([q[3] for q in test_quads],  f"{p}_nocot_ood_test.jsonl")
    write_split([q[2] for q in quads],       f"{p}_cot_ood_full.jsonl")
    write_split([q[3] for q in quads],       f"{p}_nocot_ood_full.jsonl")
    print(f"  OOD test: {len(test_quads)} pairs (perfectly aligned with IID test)")

    # Per-tier test breakdown
    print("\nTest set per tier:")
    tier_test = defaultdict(int)
    for quad in test_quads:
        tier_test[quad[0]["metadata"]["tier"]] += 1
    for t in sorted(tier_test):
        cfg = TIER_CONFIG[t]
        print(f"  Tier {t} [{cfg['nfa_min']}-{cfg['nfa_max']} states]: {tier_test[t]} examples")

    print("\nDone.")


if __name__ == "__main__":
    main()
