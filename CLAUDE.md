# FinBrief

Domain-specialised RAG assistant for equity research. The plan lives in `PLAN.md`; domain
glossary in `CONTEXT.md`; design decisions in `docs/adr/`.

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `TuringCollegeSubmissions/mrinal-AE.AFA.3.5`, via
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, label strings equal to their names. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
