"""
dfa_parallel_dataset_gen.py
============================
Generates parallel CoT / No-CoT DFA dataset from the same regex instances.

Each regex produces a 4-tuple:
  (cot_iid, nocot_iid, cot_ood, nocot_ood)

IID templates : 5 templates that mention Thompson's construction explicitly.
OOD templates : 8 paraphrased templates that do NOT mention Thompson's.

Answer format : markdown transition table (no epsilon column, -- for dead state).

4 difficulty tiers by minimised DFA state count:
  Tier 1 : 2 states      (~15%)
  Tier 2 : 2-4 states    (~25%)
  Tier 3 : 3-6 states    (~35%)
  Tier 4 : 4-18 states   (~25%)

Verification : L(NFA) == L(DFA) == L(minDFA) on all strings up to length 7.
Deduplication: MD5 hash of (regex_str, sorted_alphabet).

Output files (ChatML .jsonl):
  {prefix}_cot_{train,val,test,ood_test,full}.jsonl
  {prefix}_nocot_{train,val,test,ood_test,full}.jsonl

Usage:
  python3 dfa_parallel_dataset_gen.py \
      --n 25000 --seed 42 --out_prefix dfa --test_per_tier 200
"""

import random, json, hashlib, itertools, argparse
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, **kw): self.n=0; self.total=total
        def update(self, n=1):
            self.n+=n; print(f"\r  {self.n}/{self.total}", end="", flush=True)
        def __enter__(self): return self
        def __exit__(self, *a): print()


# ===================
#  REGEX AST
# ===================

class RegexNode: pass
class Literal(RegexNode):
    def __init__(self, c): self.char = c
class Epsilon(RegexNode): pass
class Concat(RegexNode):
    def __init__(self, l, r): self.left=l; self.right=r
class Union(RegexNode):
    def __init__(self, l, r): self.left=l; self.right=r
class Star(RegexNode):
    def __init__(self, c): self.child=c
class Plus(RegexNode):
    def __init__(self, c): self.child=c
class Optional_(RegexNode):
    def __init__(self, c): self.child=c


def desugar(node):
    if isinstance(node, (Literal, Epsilon)): return node
    if isinstance(node, Concat):    return Concat(desugar(node.left), desugar(node.right))
    if isinstance(node, Union):     return Union(desugar(node.left), desugar(node.right))
    if isinstance(node, Star):      return Star(desugar(node.child))
    if isinstance(node, Plus):      d=desugar(node.child); return Concat(d, deepcopy(Star(d)))
    if isinstance(node, Optional_): return Union(desugar(node.child), Epsilon())
    raise ValueError(f"Unknown node {node}")


def regex_to_string(node, p=0):
    if isinstance(node, Literal):   return node.char
    if isinstance(node, Epsilon):   return "eps"
    if isinstance(node, Star):
        i=regex_to_string(node.child,3)
        return f"({i})*" if isinstance(node.child,(Concat,Union)) else f"{i}*"
    if isinstance(node, Plus):
        i=regex_to_string(node.child,3)
        return f"({i})+" if isinstance(node.child,(Concat,Union)) else f"{i}+"
    if isinstance(node, Optional_):
        i=regex_to_string(node.child,3)
        return f"({i})?" if isinstance(node.child,(Concat,Union)) else f"{i}?"
    if isinstance(node, Concat):
        l=regex_to_string(node.left,2); r=regex_to_string(node.right,2)
        s=l+r; return f"({s})" if p>2 else s
    if isinstance(node, Union):
        l=regex_to_string(node.left,1); r=regex_to_string(node.right,1)
        s=f"{l}|{r}"; return f"({s})" if p>1 else s
    return "?"


# ===================
#  ALPHABETS & TIER CONFIG
# ===================

ALPHABETS = {
    "binary": ["0","1"], "ab": ["a","b"], "abc": ["a","b","c"],
    "abcd":   ["a","b","c","d"], "digits": ["0","1","2","3"],
}

TIER_CFG = {
    1: {"max_depth":4, "base_leaf":0.45, "dfa_min":2,  "dfa_max":2,
        "ops":["concat","union","concat3"], "wts":[0.55,0.35,0.10],
        "desc":"simple concat/union"},
    2: {"max_depth":5, "base_leaf":0.35, "dfa_min":2,  "dfa_max":4,
        "ops":["concat","union","star","plus","optional","concat3"],
        "wts":[0.28,0.20,0.22,0.14,0.08,0.08],
        "desc":"star/plus/optional"},
    3: {"max_depth":6, "base_leaf":0.28, "dfa_min":3,  "dfa_max":6,
        "ops":["concat","union","star","plus","concat3","star_union"],
        "wts":[0.28,0.20,0.22,0.12,0.10,0.08],
        "desc":"star_union patterns"},
    4: {"max_depth":7, "base_leaf":0.20, "dfa_min":4,  "dfa_max":18,
        "ops":["concat","union","star","plus","concat3","star_union","star_concat"],
        "wts":[0.22,0.18,0.22,0.12,0.10,0.08,0.08],
        "desc":"deeply nested closures"},
}

TIER_DIST    = {1:0.15, 2:0.25, 3:0.35, 4:0.25}
MAX_PRE_MIN  = 24   # max DFA states before minimisation


# ===================
#  RANDOM REGEX GENERATOR
# ===================

def random_regex(tier, alpha, rng, depth=0):
    cfg = TIER_CFG[tier]
    lp  = min(0.95, cfg["base_leaf"] + 0.13*depth)
    if depth >= cfg["max_depth"] or rng.random() < lp:
        return Literal(rng.choice(alpha))
    op = rng.choices(cfg["ops"], weights=cfg["wts"])[0]
    def s(): return random_regex(tier, alpha, rng, depth+1)
    if op=="concat":      return Concat(s(), s())
    if op=="concat3":     return Concat(Concat(s(),s()), s())
    if op=="union":       return Union(s(), s())
    if op=="star":        return Star(s())
    if op=="plus":        return Plus(s())
    if op=="optional":    return Optional_(s())
    if op=="star_union":  return Star(Union(s(),s()))
    if op=="star_concat": return Star(Concat(s(),s()))
    return Literal(rng.choice(alpha))


# ===================
#  THOMPSON NFA
# ===================

class NFAState:
    _ctr = 0
    def __init__(self):
        NFAState._ctr += 1; self.id=NFAState._ctr
        self.transitions=defaultdict(list); self.eps=[]
    def __repr__(self):  return f"q{self.id}"
    def __hash__(self):  return hash(self.id)
    def __eq__(self, o): return isinstance(o,NFAState) and self.id==o.id


class NFA:
    def __init__(self, s, a): self.start=s; self.accept=a
    def all_states(self):
        vis=set(); stk=[self.start]
        while stk:
            s=stk.pop()
            if s in vis: continue
            vis.add(s)
            for ts in s.transitions.values():
                stk.extend(t for t in ts if t not in vis)
            stk.extend(t for t in s.eps if t not in vis)
        return vis


def thompson(node):
    if isinstance(node, Literal):
        s,a=NFAState(),NFAState(); s.transitions[node.char].append(a); return NFA(s,a)
    if isinstance(node, Epsilon):
        s,a=NFAState(),NFAState(); s.eps.append(a); return NFA(s,a)
    if isinstance(node, Concat):
        n1=thompson(node.left); n2=thompson(node.right)
        n1.accept.eps.append(n2.start); return NFA(n1.start, n2.accept)
    if isinstance(node, Union):
        n1=thompson(node.left); n2=thompson(node.right)
        s,a=NFAState(),NFAState()
        s.eps.extend([n1.start,n2.start])
        n1.accept.eps.append(a); n2.accept.eps.append(a); return NFA(s,a)
    if isinstance(node, Star):
        n=thompson(node.child); s,a=NFAState(),NFAState()
        s.eps.extend([n.start,a]); n.accept.eps.extend([n.start,a]); return NFA(s,a)
    raise ValueError(f"thompson: unexpected {type(node)}")


# ===================
#  SUBSET CONSTRUCTION
# ===================

def eps_closure(states):
    stk=list(states); cl=set(states)
    while stk:
        s=stk.pop()
        for t in s.eps:
            if t not in cl: cl.add(t); stk.append(t)
    return frozenset(cl)


def nfa_move(states, char):
    res=set()
    for s in states: res.update(s.transitions.get(char,[]))
    return frozenset(res)


class DFA:
    def __init__(self):
        self.states=[]; self.state_index={}; self.transitions={}
        self.start=0; self.accept_states=set(); self.alphabet=[]
    def accepts(self, s):
        cur=self.start
        for c in s:
            if (cur,c) not in self.transitions: return False
            cur=self.transitions[(cur,c)]
        return cur in self.accept_states


def subset_construction(nfa, alphabet):
    dfa=DFA(); dfa.alphabet=sorted(alphabet)
    sc=eps_closure(frozenset([nfa.start]))
    dfa.states.append(sc); dfa.state_index[sc]=0
    if nfa.accept in sc: dfa.accept_states.add(0)
    queue=deque([sc])
    while queue:
        cur=queue.popleft(); ci=dfa.state_index[cur]
        for char in dfa.alphabet:
            nxt=eps_closure(nfa_move(cur,char))
            if not nxt: continue
            if nxt not in dfa.state_index:
                idx=len(dfa.states); dfa.states.append(nxt)
                dfa.state_index[nxt]=idx
                if nfa.accept in nxt: dfa.accept_states.add(idx)
                queue.append(nxt)
            dfa.transitions[(ci,char)]=dfa.state_index[nxt]
    return dfa


# ===================
#  HOPCROFT MINIMISATION
# ===================

def minimise_dfa(dfa):
    n=len(dfa.states); DEAD=n
    full={}
    for s in range(n):
        for c in dfa.alphabet: full[(s,c)]=dfa.transitions.get((s,c),DEAD)
    for c in dfa.alphabet: full[(DEAD,c)]=DEAD

    accepting     = frozenset(dfa.accept_states)
    non_accepting = frozenset(s for s in range(n) if s not in dfa.accept_states)
    dead_group    = frozenset({DEAD})
    P=set()
    if accepting:     P.add(accepting)
    if non_accepting: P.add(non_accepting)
    P.add(dead_group)

    plog=[]; step_num=0

    def pstr(partition):
        parts=[]
        for g in sorted(partition, key=lambda g:min(g)):
            labels=", ".join(f"D{s}" if s!=DEAD else "dead" for s in sorted(g))
            parts.append("{"+labels+"}")
        return "["+", ".join(parts)+"]"

    plog.append(f"Initial partition P = {pstr(P)}")

    W=set(P)
    while W:
        A=W.pop()
        for c in dfa.alphabet:
            X=frozenset(s for s in range(n+1) if full[(s,c)] in A)
            if not X: continue
            new_P=set(); changed=False
            for Y in P:
                inter=Y&X; diff=Y-X
                if inter and diff:
                    new_P.add(inter); new_P.add(diff); changed=True; step_num+=1
                    plog.append(f"  Split on '{c}': {pstr({Y})} -> {pstr({inter})} and {pstr({diff})}")
                    if Y in W: W.discard(Y); W.add(inter); W.add(diff)
                    else: W.add(inter if len(inter)<=len(diff) else diff)
                else: new_P.add(Y)
            if changed: P=new_P

    live_groups=[g for g in P if g!=dead_group]
    state_to_gi={}
    for gi,group in enumerate(live_groups):
        for s in group: state_to_gi[s]=gi

    def rep(gi): return min(s for s in live_groups[gi] if s!=DEAD)

    start_gi=state_to_gi[dfa.start]
    gi_to_new={}; q2=deque([start_gi]); vis2=set(); ctr=0
    while q2:
        gi=q2.popleft()
        if gi in vis2: continue
        vis2.add(gi); gi_to_new[gi]=ctr; ctr+=1; r=rep(gi)
        for c in dfa.alphabet:
            tgt=full[(r,c)]
            if tgt!=DEAD:
                tg=state_to_gi.get(tgt)
                if tg is not None and tg not in vis2: q2.append(tg)
    for gi in range(len(live_groups)):
        if gi not in gi_to_new: gi_to_new[gi]=ctr; ctr+=1

    min_dfa=DFA(); min_dfa.alphabet=dfa.alphabet
    min_dfa.states=list(range(len(live_groups))); min_dfa.start=0
    for gi,group in enumerate(live_groups):
        new_idx=gi_to_new[gi]; r=rep(gi)
        if any(s in dfa.accept_states for s in group if s!=DEAD):
            min_dfa.accept_states.add(new_idx)
        for c in dfa.alphabet:
            tgt=full[(r,c)]
            if tgt!=DEAD:
                tg=state_to_gi.get(tgt)
                if tg is not None: min_dfa.transitions[(new_idx,c)]=gi_to_new[tg]

    if step_num==0: plog.append("  No splits -- DFA already minimal.")
    plog.append(f"Final: {len(live_groups)} equivalence classes -> {len(live_groups)} states.")
    return min_dfa, plog


# ===================
#  VERIFICATION
# ===================

def nfa_accepts(nfa, s):
    states=eps_closure(frozenset([nfa.start]))
    for c in s: states=eps_closure(nfa_move(states,c))
    return nfa.accept in states


def verify(nfa, dfa, min_dfa, alphabet, max_len=7):
    for length in range(max_len+1):
        for combo in itertools.product(alphabet, repeat=length):
            s="".join(combo)
            a=nfa_accepts(nfa,s); b=dfa.accepts(s); c=min_dfa.accepts(s)
            if a!=b: return False, f"NFA/DFA mismatch on '{s}'"
            if a!=c: return False, f"NFA/minDFA mismatch on '{s}'"
    return True, ""


# ===================
#  DFA TABLE FORMATTER
# ===================

def fmt_D(i): return f"D{i}"


def build_dfa_table(min_dfa, alphabet):
    """Markdown transition table. No epsilon column. -- for dead/trap state."""
    sa=sorted(alphabet)
    header="| State | Role | "+" | ".join(sa)+" |"
    sep   ="|-------|------|"+"|".join(["--------"]*len(sa))+"|"
    lines =[header, sep]
    for s in range(len(min_dfa.states)):
        if s==min_dfa.start and s in min_dfa.accept_states: role="start, accept"
        elif s==min_dfa.start:                               role="start"
        elif s in min_dfa.accept_states:                     role="accept"
        else:                                                role=""
        cols=[]
        for c in sa:
            t=min_dfa.transitions.get((s,c))
            cols.append(fmt_D(t) if t is not None else "--")
        lines.append(f"| {fmt_D(s)} | {role} | "+" | ".join(cols)+" |")
    return "\n".join(lines)


def dfa_summary_line(min_dfa):
    states=", ".join(fmt_D(i) for i in range(len(min_dfa.states)))
    accept=", ".join(fmt_D(i) for i in sorted(min_dfa.accept_states))
    return (f"DFA: {len(min_dfa.states)} states ({{{states}}}), "
            f"start = {fmt_D(min_dfa.start)}, "
            f"accept = {{{accept}}}")


def sample_accepted(min_dfa, alphabet, n=3):
    found=[]
    for length in range(0,8):
        for combo in itertools.product(alphabet, repeat=length):
            word="".join(combo)
            if min_dfa.accepts(word):
                found.append(f'"{word}"' if word else '"eps"')
            if len(found)>=n: break
        if len(found)>=n: break
    return ", ".join(found) if found else "none (empty language)"


# ===================
#  COT TRACE BUILDER
# ===================

def _describe(node, depth=0):
    pad="  "*depth
    if isinstance(node,Literal):   return f"{pad}Literal '{node.char}'"
    if isinstance(node,Epsilon):   return f"{pad}epsilon"
    if isinstance(node,Star):      return f"{pad}Kleene Star (*)\n{_describe(node.child,depth+1)}"
    if isinstance(node,Plus):      return f"{pad}One-or-more (+)\n{_describe(node.child,depth+1)}"
    if isinstance(node,Optional_): return f"{pad}Optional (?)\n{_describe(node.child,depth+1)}"
    if isinstance(node,Concat):
        return f"{pad}Concatenation\n{_describe(node.left,depth+1)}\n{_describe(node.right,depth+1)}"
    if isinstance(node,Union):
        return f"{pad}Union (|)\n{_describe(node.left,depth+1)}\n{_describe(node.right,depth+1)}"
    return f"{pad}?"


def _rules(node, seen=None):
    if seen is None: seen=set()
    rules=[]; key=type(node).__name__
    if key not in seen:
        seen.add(key)
        if isinstance(node,Literal):
            rules.append(f"Symbol rule for '{node.char}': s --'{node.char}'-->  a")
        if isinstance(node,Concat):
            rules.append("Concatenation: N(r1).accept --eps--> N(r2).start")
        if isinstance(node,Union):
            rules.append("Union: new start --eps--> N(r1).start and N(r2).start; "
                         "N(r1).accept and N(r2).accept --eps--> new accept")
        if isinstance(node,Star):
            rules.append("Kleene Star: new start s --eps--> N(r).start (enter), "
                         "s --eps--> a (skip); N(r).accept --eps--> N(r).start (loop), "
                         "N(r).accept --eps--> a (exit)")
    children=([node.left,node.right] if isinstance(node,(Concat,Union))
               else [node.child] if isinstance(node,(Star,Plus,Optional_)) else [])
    for c in children: rules+=_rules(c,seen)
    return rules


def build_cot_response(regex_string, ast_raw, nfa, dfa, min_dfa, plog, alphabet):
    sa=sorted(alphabet)
    nfa_states=sorted(nfa.all_states(), key=lambda s:s.id)
    lines=[]

    # Step 1
    lines += [
        "## Step 1: Analyse the Regular Expression\n",
        f"Given regex: {regex_string}",
        f"Alphabet S = {{{', '.join(sa)}}}\n",
        "Parse tree structure:",
        _describe(ast_raw),
    ]

    # Step 2
    lines += ["","## Step 2: Thompson's Construction Rules Applied\n"]
    for r in _rules(ast_raw):
        lines.append(f"  - {r}")

    # Step 3: NFA
    lines += [
        "","## Step 3: NFA States and Transitions\n",
        f"NFA states: {{{', '.join(f'q{s.id}' for s in nfa_states)}}}",
        f"Start state : q{nfa.start.id}",
        f"Accept state: q{nfa.accept.id}\n",
    ]
    for s in nfa_states:
        for char,targets in sorted(s.transitions.items()):
            tstr="{"+", ".join(f"q{t.id}" for t in sorted(targets,key=lambda x:x.id))+"}"
            lines.append(f"  q{s.id} --'{char}'--> {tstr}")
        if s.eps:
            tstr="{"+", ".join(f"q{t.id}" for t in sorted(s.eps,key=lambda x:x.id))+"}"
            lines.append(f"  q{s.id} --eps--> {tstr}")

    # Step 4: eps-closure of start
    sc=eps_closure(frozenset([nfa.start]))
    ids=sorted(x.id for x in sc)
    lines += [
        "","## Step 4: Compute eps-closure of Start State\n",
        f"eps-closure(q{nfa.start.id}) = {{{', '.join(f'q{i}' for i in ids)}}}",
        "This becomes DFA start state D0.\n",
    ]

    # Step 5: subset construction table
    lines += ["","## Step 5: Subset Construction (Powerset Construction)\n"]
    hdr  = "| DFA State | NFA States | "+" | ".join(sa)+" | Accept? |"
    sepr = "|-----------|------------|"+"|".join(["--------"]*len(sa))+"|---------|"
    lines += [hdr, sepr]
    for i,nfa_set in enumerate(dfa.states):
        row=[]
        for c in sa:
            t=dfa.transitions.get((i,c))
            row.append(fmt_D(t) if t is not None else "empty")
        nfa_ids="{"+", ".join(f"q{s.id}" for s in sorted(nfa_set,key=lambda x:x.id))+"}"
        acc="Y" if i in dfa.accept_states else ""
        lines.append(f"| {fmt_D(i)} | {nfa_ids} | "+" | ".join(row)+f" | {acc} |")
    lines += [
        "",
        f"Total DFA states before minimisation: {len(dfa.states)}",
        f"Accept states: {{{', '.join(fmt_D(i) for i in sorted(dfa.accept_states))}}}\n",
    ]

    # Step 6: Hopcroft
    lines += [
        "## Step 6: DFA Minimisation (Hopcroft's Procedure)\n",
        "Partition refinement trace:\n",
    ]
    for entry in plog: lines.append(entry)
    lines.append("")
    if len(min_dfa.states)==len(dfa.states):
        lines.append("DFA is already minimal -- all states are distinguishable.\n")
    else:
        lines.append(f"Minimisation reduced {len(dfa.states)} -> {len(min_dfa.states)} states.\n")

    # Step 7: Final minimised DFA table (THE SHARED ANSWER)
    lines += [
        "## Step 7: Final Minimised DFA\n",
        f"States Q = {{{', '.join(fmt_D(i) for i in range(len(min_dfa.states)))}}}",
        f"Start state: {fmt_D(min_dfa.start)}",
        f"Accept states F = {{{', '.join(fmt_D(i) for i in sorted(min_dfa.accept_states))}}}",
        f"Alphabet S = {{{', '.join(sa)}}}\n",
        "Transition function (-- = dead/trap state transition):\n",
        build_dfa_table(min_dfa, alphabet),
        "",
        dfa_summary_line(min_dfa),
    ]

    # Step 8: Verification
    lines += [
        "","## Step 8: Verification\n",
        f"Example accepted strings: {sample_accepted(min_dfa, alphabet)}",
        f"The regex {regex_string} is recognised by a minimised DFA with "
        f"{len(min_dfa.states)} state(s).",
    ]

    return "\n".join(lines)


def build_nocot_response(min_dfa, alphabet):
    """No-CoT: final DFA table only. Identical to Step 7 output of CoT."""
    lines=[
        "## DFA Transition Table (Minimised)\n",
        "The following table shows the minimised DFA. "
        "'--' denotes a dead/trap state transition.\n",
        build_dfa_table(min_dfa, alphabet),
        "",
        dfa_summary_line(min_dfa),
    ]
    return "\n".join(lines)


# ===================
#  QUESTION TEMPLATES
# ===================

SYSTEM_PROMPT = (
    "You are an expert in formal language theory and automata. "
    "When given a regular expression, you convert it to a minimised "
    "Deterministic Finite Automaton (DFA) using Thompson's construction, "
    "subset (powerset) construction, and Hopcroft's minimisation algorithm. "
    "You present the final DFA as a clear transition table."
)

IID_TEMPLATES = [
    "Convert the regular expression `{regex}` over the alphabet S = {{{alpha}}} "
    "to a minimised DFA using Thompson's construction. Show all steps.",
    "Given the regex `{regex}` with alphabet {{{alpha}}}, construct a minimised "
    "DFA using Thompson's construction followed by subset construction. "
    "Show your full work.",
    "Use Thompson's NFA construction and the powerset construction to build a "
    "DFA for the regex `{regex}` (S = {{{alpha}}}), then minimise it using "
    "Hopcroft's procedure.",
    "Construct a minimised DFA that accepts exactly the language described by "
    "the regular expression `{regex}` over S = {{{alpha}}}. Show the NFA, "
    "DFA, and minimised DFA.",
    "For the regular expression `{regex}` over alphabet {{{alpha}}}, perform:\n"
    "1. Thompson's NFA construction\n"
    "2. Subset construction to DFA\n"
    "3. DFA minimisation using Hopcroft's procedure\n"
    "Show all intermediate steps.",
]

OOD_TEMPLATES = [
    "I need a DFA for the pattern {regex} using symbols {{{alpha}}}. "
    "Please show me the complete state transition table.",
    "Design a deterministic finite automaton that recognizes the language "
    "defined by {regex} over {{{alpha}}}. List all states, mark the start "
    "and accepting states, and give the complete transition table.",
    "What DFA would you build to recognize strings matching {regex} where "
    "the alphabet is {{{alpha}}}? Show the transition table.",
    "For the regular expression {regex} with input alphabet {{{alpha}}}, "
    "construct a minimal deterministic finite automaton. Present the states "
    "and how they transition on each input symbol.",
    "Exercise: Given r = {regex} over alphabet {{{alpha}}}, construct the "
    "minimal DFA M such that L(M) = L(r). Indicate start and accepting states.",
    "I want to implement a regex matcher for {regex} on the alphabet {{{alpha}}}. "
    "Build me the minimal DFA -- show every state and every transition.",
    "Suppose we want to check if strings over {{{alpha}}} match the pattern "
    "{regex}. What minimal DFA would do this? Give the complete transition "
    "function as a table.",
    "Let r = {regex} be a regular expression over Sigma = {{{alpha}}}. "
    "Construct a minimal DFA M = (Q, Sigma, delta, q0, F) such that "
    "L(M) = L(r). Specify Q, q0, F, and delta in tabular form.",
]


# ===================
#  HANDCRAFTED EXAMPLES
# ===================

HANDCRAFTED = [
    ("(a|b)*abb",           ["a","b"]),
    ("(0|1)*00",            ["0","1"]),
    ("(0|1)*101",           ["0","1"]),
    ("(a|b)*ab",            ["a","b"]),
    ("(a|b)*bab",           ["a","b"]),
    ("(0|1)*11(0|1)*",      ["0","1"]),
    ("a*b*",                ["a","b"]),
    ("a*ba*",               ["a","b"]),
    ("(aa|b)*",             ["a","b"]),
    ("(ab|ba)*",            ["a","b"]),
    ("a+b+",                ["a","b"]),
    ("(ab)+",               ["a","b"]),
    ("(a|b)*aa(a|b)*",      ["a","b"]),
    ("(0|1)*00(0|1)*",      ["0","1"]),
    ("(a|b)(a|b)(a|b)",     ["a","b"]),
    ("(aa)*",               ["a","b"]),
    ("(0|1)*0(0|1)",        ["0","1"]),
    ("(a|b)*aba(a|b)*",     ["a","b"]),
    ("(0|1)*001(0|1)*",     ["0","1"]),
    ("(0|1)*(00|11)(0|1)*", ["0","1"]),
]


# ===================
#  REGEX PARSER
# ===================

class _Parser:
    def __init__(self, s): self.s=s; self.pos=0
    def peek(self): return self.s[self.pos] if self.pos<len(self.s) else None
    def consume(self, c=None):
        ch=self.s[self.pos]
        if c and ch!=c: raise ValueError(f"Expected {c!r} got {ch!r}")
        self.pos+=1; return ch
    def parse(self):
        node=self._union()
        if self.pos!=len(self.s): raise ValueError(f"Leftover at {self.pos}")
        return node
    def _union(self):
        left=self._concat()
        while self.peek()=='|': self.consume('|'); left=Union(left,self._concat())
        return left
    def _concat(self):
        nodes=[]
        while self.peek() not in (None,')','|'): nodes.append(self._quantified())
        if not nodes: raise ValueError("Empty concat")
        result=nodes[0]
        for n in nodes[1:]: result=Concat(result,n)
        return result
    def _quantified(self):
        base=self._atom(); q=self.peek()
        if q=='*': self.consume(); return Star(base)
        if q=='+': self.consume(); return Plus(base)
        if q=='?': self.consume(); return Optional_(base)
        return base
    def _atom(self):
        c=self.peek()
        if c=='(':
            self.consume('('); node=self._union(); self.consume(')'); return node
        if c and c not in (')','|','*','+','?'):
            self.consume(); return Literal(c)
        raise ValueError(f"Unexpected char {c!r}")

def parse_regex(s): return _Parser(s).parse()

def _guess_tier(regex_string):
    n=len(regex_string)
    has_closure = any(c in regex_string for c in ('+','*','?'))
    if n<=5 and not has_closure: return 1
    if n<=12: return 2
    if n<=22: return 3
    return 4


# ===================
#  ENTRY BUILDER
# ===================

def _build_quad(regex_string, ast_raw, alphabet, rng, tier):
    NFAState._ctr = 0
    try:
        ast     = desugar(ast_raw)
        nfa     = thompson(ast)
        dfa     = subset_construction(nfa, alphabet)
        if len(dfa.states) > MAX_PRE_MIN: return None
        min_dfa, plog = minimise_dfa(dfa)
    except Exception:
        return None

    cfg   = TIER_CFG[tier]
    n_min = len(min_dfa.states)
    if not (cfg["dfa_min"] <= n_min <= cfg["dfa_max"]): return None

    ok, _ = verify(nfa, dfa, min_dfa, alphabet, max_len=7)
    if not ok: return None

    alpha_str   = ", ".join(sorted(alphabet))
    iid_q       = rng.choice(IID_TEMPLATES).format(regex=regex_string, alpha=alpha_str)
    ood_q       = rng.choice(OOD_TEMPLATES).format(regex=regex_string, alpha=alpha_str)
    cot_resp    = build_cot_response(regex_string, ast_raw, nfa, dfa, min_dfa, plog, alphabet)
    nocot_resp  = build_nocot_response(min_dfa, alphabet)

    meta = {
        "regex":          regex_string,
        "alphabet":       sorted(alphabet),
        "tier":           tier,
        "nfa_states":     len(nfa.all_states()),
        "dfa_states":     len(dfa.states),
        "min_dfa_states": n_min,
        "accept_states":  sorted(min_dfa.accept_states),
        "hash":           hashlib.md5((regex_string+str(sorted(alphabet))).encode()).hexdigest()[:8],
    }

    def entry(resp, question):
        return {"messages":[
            {"role":"system",    "content":SYSTEM_PROMPT},
            {"role":"user",      "content":question},
            {"role":"assistant", "content":resp},
        ], "metadata":meta}

    return (
        entry(cot_resp,   iid_q),   # cot_iid
        entry(nocot_resp, iid_q),   # nocot_iid
        entry(cot_resp,   ood_q),   # cot_ood
        entry(nocot_resp, ood_q),   # nocot_ood
    )


# ===================
#  GENERATION PIPELINE
# ===================

def generate_handcrafted(rng):
    quads=[]
    for regex_string, alphabet in HANDCRAFTED:
        try: ast_raw=parse_regex(regex_string)
        except Exception as e:
            print(f"  [WARN] Parse failed '{regex_string}': {e}"); continue
        result=_build_quad(regex_string, ast_raw, alphabet, rng, _guess_tier(regex_string))
        if result: quads.append(result)
        else: print(f"  [WARN] Build failed for '{regex_string}'")
    return quads


def generate_random_quad(rng, tier):
    alpha    = ALPHABETS[rng.choice(list(ALPHABETS.keys()))]
    ast_raw  = random_regex(tier, alpha, rng)
    rstr     = regex_to_string(ast_raw)
    if len(rstr)<2: return None
    return _build_quad(rstr, ast_raw, alpha, rng, tier)


def print_stats(quads):
    print("\n"+"="*65)
    print("  DATASET STATISTICS")
    print("="*65)
    tc={}; sc=[]
    for q in quads:
        m=q[0]["metadata"]; t=m["tier"]
        tc[t]=tc.get(t,0)+1; sc.append(m["min_dfa_states"])
    print(f"  Total quads    : {len(quads):,}")
    for t in sorted(tc):
        cfg=TIER_CFG[t]
        print(f"  Tier {t} [{cfg['dfa_min']}-{cfg['dfa_max']} states] {cfg['desc']:30s}: {tc[t]:,}")
    print(f"  minDFA states  : avg={sum(sc)/len(sc):.1f}, min={min(sc)}, max={max(sc)}")
    print("="*65+"\n")


def write_jsonl(data, path):
    with open(path,"w",encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex,ensure_ascii=False)+"\n")
    print(f"  Wrote {len(data):,} -> {path}")


# ===================
#  MAIN
# ===================

def main():
    ap=argparse.ArgumentParser(description="Generate parallel CoT/No-CoT DFA datasets")
    ap.add_argument("--n",             type=int,   default=25000)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--out_prefix",    type=str,   default="dfa")
    ap.add_argument("--val_split",     type=float, default=0.05)
    ap.add_argument("--test_per_tier", type=int,   default=200)
    args=ap.parse_args()

    rng=random.Random(args.seed); quads=[]; seen=set()

    print("Generating handcrafted examples...")
    for q in generate_handcrafted(rng):
        h=q[0]["metadata"]["hash"]
        if h not in seen: seen.add(h); quads.append(q)
    print(f"  Added {len(quads)} handcrafted quads.")

    print(f"Generating up to {args.n:,} random quads...")
    max_attempts=args.n*40
    with tqdm(total=args.n) as pbar:
        pbar.update(len(quads)); attempts=0
        while len(quads)<args.n and attempts<max_attempts:
            attempts+=1
            tier=rng.choices(list(TIER_DIST),weights=list(TIER_DIST.values()))[0]
            result=generate_random_quad(rng,tier)
            if result is None: continue
            h=result[0]["metadata"]["hash"]
            if h in seen: continue
            seen.add(h); quads.append(result); pbar.update(1)

    print(f"\nGenerated {len(quads):,} quads in {attempts:,} attempts.")
    print_stats(quads)

    # Stratified test split (args.test_per_tier per tier)
    rng.shuffle(quads)
    by_tier=defaultdict(list)
    for q in quads: by_tier[q[0]["metadata"]["tier"]].append(q)

    test_q=[]; train_q=[]
    for tier in sorted(by_tier):
        tier_list=by_tier[tier]
        n_test=min(args.test_per_tier, len(tier_list)//5)
        test_q.extend(tier_list[:n_test])
        train_q.extend(tier_list[n_test:])

    rng.shuffle(train_q)
    n_val=int(len(train_q)*args.val_split)
    val_q=train_q[:n_val]; train_q=train_q[n_val:]

    p=args.out_prefix
    print(f"\nSplit: train={len(train_q):,} | val={len(val_q):,} | test={len(test_q):,}")

    # IID splits
    for split,data in [("train",train_q),("val",val_q),("test",test_q)]:
        write_jsonl([q[0] for q in data], f"{p}_cot_{split}.jsonl")
        write_jsonl([q[1] for q in data], f"{p}_nocot_{split}.jsonl")

    # OOD test split (aligned with IID test -- same regex instances)
    write_jsonl([q[2] for q in test_q], f"{p}_cot_ood_test.jsonl")
    write_jsonl([q[3] for q in test_q], f"{p}_nocot_ood_test.jsonl")

    # Full datasets
    write_jsonl([q[0] for q in quads], f"{p}_cot_full.jsonl")
    write_jsonl([q[1] for q in quads], f"{p}_nocot_full.jsonl")

    print(f"\n  OOD test: {len(test_q)} pairs (perfectly aligned with IID test)")
    print("\nPer-tier test breakdown:")
    tc=defaultdict(int)
    for q in test_q: tc[q[0]["metadata"]["tier"]]+=1
    for t in sorted(tc):
        cfg=TIER_CFG[t]
        print(f"  Tier {t} [{cfg['dfa_min']}-{cfg['dfa_max']} states]: {tc[t]} examples")
    print("\nDone.")

if __name__=="__main__":
    main()
