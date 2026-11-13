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
    scores        jsonb       NOT NULL,   -- {model: {p, c}}: P(pass) and price for every eligible model
    shadow_model  text,                   -- extra render requested for learning, never shown to the customer
    shadow_propensity real
);
CREATE INDEX IF NOT EXISTS decisions_created ON decisions (created_at);

-- one row per rendered model: the routed render, plus any shadow render for the same decision
CREATE TABLE IF NOT EXISTS feedback (
    decision_id     uuid        NOT NULL,
    model           text        NOT NULL,
    received_at     timestamptz NOT NULL DEFAULT now(),
    passed          boolean     NOT NULL,
    credits_charged real,
    shadow          boolean     NOT NULL DEFAULT false,
    PRIMARY KEY (decision_id, model)
);

-- migrations for databases created before shadow exploration
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS shadow_model text;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS shadow_propensity real;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS shadow boolean NOT NULL DEFAULT false;
DO $$ BEGIN
    IF (SELECT count(*) FROM information_schema.key_column_usage
        WHERE table_name = 'feedback' AND constraint_name = 'feedback_pkey') = 1 THEN
        ALTER TABLE feedback DROP CONSTRAINT feedback_pkey;
        ALTER TABLE feedback ADD PRIMARY KEY (decision_id, model);
    END IF;
END $$;
