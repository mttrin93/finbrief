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
