# AWS and Dokploy staging deployment

This runbook deploys the backend as a restricted staging environment. The React
frontend remains on AWS Amplify. Dokploy builds this GitHub repository with
`docker-compose.dokploy.yml` and runs PostgreSQL, the migration job, API, and
worker on one EC2 instance.

The fixed email and access code are a temporary staging gate. They are not a
replacement for production identity, per-user authorization, or abuse controls.

## AWS foundation

1. Use `me-central-1` unless the existing Amplify and DNS setup requires another
   region. Start with an Ubuntu `t3.large`, 50 GB encrypted gp3 root volume, an
   Elastic IP, and automated EC2 snapshots. Dokploy and PostgreSQL share the host
   and need memory headroom even though AI requests run on external providers.
2. Allow inbound `80` and `443` publicly. Restrict `22` and Dokploy's `3000` UI
   to administrator IPs or a private VPN. Do not expose PostgreSQL or port `8000`.
3. Create a private S3 bucket with Block Public Access, default encryption,
   versioning, and a lifecycle policy. Attach an EC2 role limited to the asset
   prefix with `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, and
   `s3:ListBucket`.
4. Require IMDSv2. If containers cannot retrieve role credentials, set the
   metadata response hop limit to `2`; do not add AWS keys to the app environment.
5. Point the API DNS `A` record at the Elastic IP. In Dokploy, add the domain to
   the Compose `api` service on port `8000` and enable HTTPS. Dokploy supplies the
   Traefik labels, so the Compose file does not publish the port.

## Dokploy project

1. Connect `m-elkhamisy/AI-Launch-Kit-Backend` through GitHub and select the
   release branch or commit under test.
2. Create a standard Docker Compose service using
   `docker-compose.dokploy.yml` and isolated deployments. Enable auto-deploy only
   after CI is required on the release branch.
3. Add the environment below in Dokploy. Generate independent URL-safe values
   for the database password, webhook tokens, and authentication token secret.

```dotenv
POSTGRES_DB=launchkit
POSTGRES_USER=launchkit
POSTGRES_PASSWORD=<url-safe-random-password>

LAUNCHKIT_ENVIRONMENT=staging
LAUNCHKIT_DEBUG=false
LAUNCHKIT_LOG_LEVEL=INFO
LAUNCHKIT_LOG_JSON=true
LAUNCHKIT_DATABASE_URL=postgresql+asyncpg://launchkit:<url-encoded-password>@postgres:5432/launchkit
LAUNCHKIT_DATABASE_ECHO=false

LAUNCHKIT_AUTH_MODE=oauth
LAUNCHKIT_AUTH_EMAIL=
LAUNCHKIT_AUTH_OTP=
LAUNCHKIT_AUTH_TOKEN_SECRET=<at-least-32-random-bytes>
LAUNCHKIT_AUTH_TOKEN_TTL_SECONDS=28800

LAUNCHKIT_FRONTEND_ORIGINS=https://ai-launch-kitt-git-codex-aws-dokploy-readiness-innovation-city.vercel.app,https://<amplify-production-origin>
LAUNCHKIT_SITE_URL=https://<api-domain>
LAUNCHKIT_CLAIM_RETURN_URL=https://<amplify-production-origin>/

LAUNCHKIT_S3_BUCKET=<private-asset-bucket>
LAUNCHKIT_S3_ASSET_PREFIX=staging/assets/
LAUNCHKIT_AWS_REGION=me-central-1

LAUNCHKIT_OPENROUTER_API_KEY=<secret>
LAUNCHKIT_V0_API_KEY=<secret>
LAUNCHKIT_V0_WEBHOOK_TOKEN=<random-secret>
LAUNCHKIT_V0_WEBHOOK_CALLBACK_URL=https://<api-domain>/api/v1/webhooks/v0/<same-random-secret>
LAUNCHKIT_VERCEL_TOKEN=<secret>
LAUNCHKIT_VERCEL_TEAM_ID=
LAUNCHKIT_VERCEL_WEBHOOK_SECRET=<secret>
```

Retain the remaining model, pacing, size, and reconciliation defaults from
`.env.example`. Keep `LAUNCHKIT_VERCEL_TEAM_ID` empty for a personal Vercel
account; a real team value must start with `team_`.

## Release and callbacks

1. Preview the rendered Compose file. Confirm only `api` has a public domain,
   PostgreSQL has a named volume, and no secret appears in build arguments.
2. Deploy. PostgreSQL must become healthy and `migrate` must exit `0` before the
   API and worker start.
3. Verify `https://<api-domain>/api/v1/health` and `/api/v1/ready` return `200`.
4. In the API terminal, run `python -m launchkit.hooks provision-v0`. Running it
   twice must be idempotent.
5. Configure the Vercel webhook at
   `https://<api-domain>/api/v1/webhooks/vercel` with the matching secret.
6. In Amplify, set `VITE_API_BASE_URL=https://<api-domain>` and redeploy. This is
   public configuration, not a secret.

## Backups, monitoring, and rollback

- Configure a nightly Dokploy PostgreSQL backup to S3, retain at least 14 copies,
  run a manual backup immediately, and test one restore before accepting data.
  Back up Dokploy itself separately.
- Monitor EC2 checks, CPU, memory, disk, API readiness, worker restarts, failed
  jobs, and provider `429`/`5xx` rates. Alert before disk reaches 80%.
- Keep API and worker at one replica while PostgreSQL and provider pacing are on
  one host. Scale only after shared limits and observability move out of process.
- Roll back by redeploying the previous tested Git commit. Keep Alembic changes
  backward compatible across one release; restore PostgreSQL only for data loss
  or a migration incident.

## Staging acceptance

Log in through InnovationCity OAuth (WeCan-registered client), complete one full workflow,
refresh during processing, download the ZIP, and create a Vercel claim deployment.
Confirm a missing or expired token returns `401`. Record only internal IDs and
sanitized logs, never provider references, tokens, claim codes, or customer data.
