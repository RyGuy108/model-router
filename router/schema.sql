-- Postgres is the source of truth for decisions, outcomes and model versions.
-- Redis only holds rebuildable, hot state (decision cache, live cost / bias tables, model pubsub).

CREATE TABLE IF NOT EXISTS model_registry (
    version     text PRIMARY KEY,
    artifact    bytea       NOT NULL,
    tau         jsonb       NOT NULL,
    metrics     jsonb       NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    active      boolean     NOT NULL DEFAULT false
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_model ON model_registry (active) WHERE active;

CREATE TABLE IF NOT EXISTS decisions (
    id            uuid PRIMARY KEY,
    tenant        text        NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    model_version text        NOT NULL,
    request       jsonb       NOT NULL,
    chosen        text        NOT NULL,
    p_pass        real        NOT NULL,
    est_credits   real        NOT NULL,
    propensity    real        NOT NULL,   -- P(policy picks `chosen` | request); < 1 when exploring
    explored      boolean     NOT NULL,
    reason        text        NOT NULL,
    scores        jsonb       NOT NULL    -- {model: p_pass} for every eligible model
);
CREATE INDEX IF NOT EXISTS decisions_created ON decisions (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    decision_id     uuid PRIMARY KEY,
    received_at     timestamptz NOT NULL DEFAULT now(),
    model           text        NOT NULL,
    passed          boolean     NOT NULL,
    credits_charged real
);
