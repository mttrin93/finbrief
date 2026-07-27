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
`search_filings` offsets each result by the number of sources the thread has already issued —
counted from the thread's own tool messages, which the checkpointer already persists. No new
state, and the register cannot drift from the transcript it numbers. This is the decision's
"the agent's context always comes from the checkpointer" doing more work than it was written
for, and it is why the sequence is deliberately *thread-global* rather than per-turn: `[7]` then
means one chunk for a whole conversation, so a marker in an earlier answer still resolves to the
source the reader was shown beside it (ADR-0003 amendment §3).

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

**Also true, and outside the decision as written.** The `thread_id` is logged with every turn
(`agent_query`, `agent_turn`): it is a conversation identifier, not user content, and grouping
turns by conversation is what lets T10 (#11) report per-conversation behaviour. `session_state`
also holds one flag per assistant row — whether the turn searched — which is display data, since
"the agent answered from the conversation" and "the collection returned nothing" render
differently and only the second is a setup problem.
