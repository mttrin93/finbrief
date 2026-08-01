# ADR-0008: Conversation-state ownership across the agent and Streamlit

`create_agent`'s checkpointer and Streamlit's rerun model are two candidate sources of
truth for the conversation. Streamlit re-runs the whole script on every interaction, so a
checkpointer built per-run remembers nothing, a fresh-per-run `thread_id` loses memory,
and anything cached at module level is shared across all user sessions (a privacy leak on
a deployed multi-user app).

**Decision.**

- **The checkpointer is the agent's memory of record** — `SqliteSaver`, file-backed. The
  agent's context always comes from the checkpointer.
- **`st.session_state` holds only** the `thread_id`, UI toggles (model, strategy), and the
  display transcript. It is never the agent's memory.
- **Agent + checkpointer are built once under `@st.cache_resource`.** The SQLite connection
  is created with `check_same_thread=False`, because Streamlit reruns land on different
  threads.
- **`thread_id` = `uuid4` minted once per session** into `session_state`: stable across
  reruns, unique across sessions. This isolates users on the shared cached instance.
  Tier-2 auth re-keys `thread_id` by user id.

**Stated semantics.**
- Browser refresh starts a new conversation (`session_state` reset → new uuid; old threads
  orphan harmlessly).
- The checkpointer file is ephemeral on Streamlit Community Cloud — accepted; conversations
  are not treated as durable data.

**Enforcement.** `test_app_state.py` (streamlit.testing.v1.AppTest) asserts: `thread_id`
stability across reruns; distinctness across two AppTest sessions; fresh-uuid + surviving
cached agent on start-over. This is where the prior review's session-state-testing
recommendation is implemented.

---

## Amendment (ticket T4, issue #7) — what implementing it added

The decision above stands as written. Four things it did not say, recorded here rather than
left in the code for the next reader to reconstruct.

**1. The checkpointer is also the citation register.** `retrieve()` ranks 1…k on every call, so
a thread's search replies are renumbered into one running sequence, computed from the thread's own
tool messages — which the checkpointer already persists. No new state, and the register cannot
drift from the transcript it numbers. This is the decision's "the agent's context always comes
from the checkpointer" doing more work than it was written for, and it is why the sequence is
deliberately *thread-global* rather than per-turn: `[7]` then means one chunk for a whole
conversation, so a marker in an earlier answer still resolves to the source the reader was shown
beside it (ADR-0003 amendment §3).

**Where** the renumbering runs is not a detail. T4 first did it inside `search_filings`, which
cannot be correct: LangGraph builds every `ToolRuntime` in a step from the same state and then
runs the calls concurrently, so two searches in one step read the same "already issued" count and
both emit `[1…k]`. It now runs at the `before_model` seam (`agent/citations.py`) as one sequential
pass — the only reader of the register is also its only writer, which is what makes a collision
unrepresentable rather than merely unlikely (issue #7 review; ADR-0003 amendment §3 and §5). The
pass is idempotent and recomputes the whole sequence, so a thread checkpointed by an earlier
shape is numbered correctly the next time it is read rather than needing a migration.

**2. Nothing that crosses into the checkpoint may be a domain object.** Every message is
serialised, artifacts included. A frozen dataclass holding an enum survives that round trip only
through an escape hatch LangGraph warns on and will remove, and comes back as an untyped dict
under its strict serialiser — silently. So `search_filings` returns `Context.as_payload()` dicts
and the agent rebuilds them for display. No cost in checkpoint size: the tool message's *content*
already carries those bodies, because that is what the model reads.

**3. "Start over" mints a thread; it does not clear the checkpointer.** The old thread is
orphaned rather than deleted — nothing can reach it, and it is ephemeral anyway. Clearing the
saver, or rebuilding the cached agent, would discard **every other session's** memory, which is
precisely the cross-user failure this ADR exists to prevent, arriving through the button that
looks like the safe one. `test_starting_over_does_not_reach_another_session` holds it down.

**4. The stated semantics have to be stated *on screen*.** "Browser refresh starts a new
conversation" is an accepted consequence, so the sidebar says so — a user who is not told reads
a lost conversation as a bug, and a reviewer cannot tell an accepted consequence from an
oversight. `test_the_page_states_that_a_refresh_starts_a_new_conversation` binds it.

---

## Amendment (ticket T14, issue #15) — one cached agent becomes one per model

The decision above stands, and the model picker it already anticipated ("UI toggles (model,
strategy)") is now built. Two things it did not say, one of them a real change to the third
bullet.

**1. `model` is a cache key, and isolation does not come from there.** `shared_agent(model)` is
still `@st.cache_resource`d, which keys on the arguments — so it yields **one agent per model a
reader has picked** and replaces nothing: the entry for the previous model stays live for the
sessions still on it. **Measured before it was built**, because this is the mechanism the
two-session guarantee sits on and a *replacement* would have handed one session another's agent
mid-turn:

```
builds   : ['openai/gpt-4o-mini', 'anthropic/claude-3.5-haiku']   # two, not three
a1 is a3 : True    # the first survives after the second is built
a1 is b1 : False   # distinct instances
```

Isolation is therefore **unchanged**, and the reason is the fourth bullet above rather than the
third: it was carried by the per-session `uuid4` `thread_id` all along and never by there being
one agent.

**That is also why the isolation test cannot be the regression guard for the picker**, and the
point is worth stating because it is this repo's recurring bug class. Drop `model` from the
cached builder and every session silently answers on `FINBRIEF_CHAT_MODEL` — while still holding
two distinct thread ids, still sending each turn to its own. An isolation test passes before and
after. So `test_app_state.py` carries **two** tests: one for the property (two sessions on two
models stay separate) and one that detects the mutation (the picked model reaches `build_agent`,
as an equality on the built models). Both mutations were applied and the second failed on both
while the first passed on both — measured, not reasoned about.

**2. The process now opens up to four SQLite connections where it opened one.** Every
`build_agent` resolves `build_checkpointer(settings)` to the same `checkpoint_db`, so one cache
entry per model is one `sqlite3.connect` per model picked. This is a genuine change to "Agent +
checkpointer are built once", and it is accepted rather than worked around — because it is the
same fact that makes a **conversation survive a model switch**: the file and the `thread_id` are
what a conversation is, so a second agent reading the same file under the same thread is handed
the first's history. Measured, at the seam (`test_agent.py`), asserting that the second model was
*shown* the earlier turn and not merely that the state accumulated.

What contention looks like if it ever bites: two sessions on two models writing concurrently now
contend across connections where they previously serialised behind one `SqliteSaver` lock.
`sqlite3.connect`'s default 5-second busy timeout absorbs the overlap — a checkpoint write is
milliseconds — and past it a turn raises `database is locked` from inside `answer()`, which
`app/Home.py`'s generic error branch renders. Nothing corrupts: SQLite's own file locking is what
is being relied on, which is the same promise `check_same_thread=False` already leans on. If it
does bite, the fix is one shared checkpointer injected into every `build_agent` rather than one
resolved per build — deliberately **not** done pre-emptively, because it trades a measured
non-problem for a second cached singleton on a path that already has six.

**Also true, and outside the decision as written.** The `thread_id` is logged with every turn
(`agent_query`, `agent_turn`): it is a conversation identifier, not user content, and grouping
turns by conversation is what lets T10 (#11) report per-conversation behaviour. `session_state`
also holds one flag per assistant row — whether the turn searched — which is display data, since
"the agent answered from the conversation" and "the collection returned nothing" render
differently and only the second is a setup problem.
