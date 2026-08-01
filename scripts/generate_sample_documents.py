"""
Sample Document Generator — Generates a diverse, realistic corpus of synthetic
corporate policy, SOP, code, and configuration files for evaluation testing.

Covers all major supported file types: Markdown, plain text, CSV, TSV, JSONL,
Python, SQL, Shell script, JSON, YAML, and HTML.
"""

import os
from pathlib import Path

SAMPLE_DOCS_DIR = "data/sample_documents"

# ---------------------------------------------------------------------------
# Markdown Documents
# ---------------------------------------------------------------------------

MD_VACATION_POLICY = (
    "company_vacation_policy.md",
    """# Acme Corp Employee Vacation & Paid Time Off (PTO) Policy

## 1. Overview
Acme Corp provides eligible employees with Paid Time Off (PTO) for vacation, personal affairs, and illness. This policy applies to all full-time employees.

## 2. PTO Accrual Rate
- **0 to 2 years of service**: 15 days of PTO per calendar year (1.25 days per month).
- **2 to 5 years of service**: 20 days of PTO per calendar year (1.66 days per month).
- **5+ years of service**: 25 days of PTO per calendar year (2.08 days per month).

## 3. Rollover Policy
Employees may roll over a maximum of 5 unused PTO days into the following calendar year. Any additional unused days beyond 5 days are forfeited on December 31st.

## 4. Request Procedure
PTO requests must be submitted through the internal HR Portal at least 2 weeks in advance for planned leave exceeding 3 consecutive business days.

## 5. Blackout Periods
PTO requests during fiscal year-end (December 15–January 5) and company-wide release windows require VP-level approval. Blackout periods are announced at least 30 days in advance.

## 6. Emergency Leave
Employees may take up to 3 days of unplanned emergency leave per quarter without prior notice. Documentation (medical certificate or equivalent) must be provided within 5 business days.
""",
)

MD_ONCALL_SOP = (
    "engineering_oncall_sop.md",
    """# Engineering On-Call Escalation SOP

## 1. Incident Classification
- **P1 (Critical)**: Production service complete outage or data corruption. SLA resolution target: 1 hour.
- **P2 (High)**: Major feature broken impacting > 25% users. SLA resolution target: 4 hours.
- **P3 (Medium)**: Minor bug with available workaround. SLA resolution target: 24 hours.
- **P4 (Low)**: Cosmetic issues or feature requests. SLA resolution target: 5 business days.

## 2. On-Call Rotation Schedule
The primary on-call engineer shifts every Monday at 09:00 AM UTC. A secondary backup engineer is assigned to handle escalations if the primary fails to acknowledge a P1 alert within 15 minutes.

## 3. PagerDuty Alert Response
1. Acknowledge alert in PagerDuty within 15 minutes for P1 incidents.
2. Open a dedicated Slack incident channel (`#inc-YYYYMMDD-description`).
3. Post status update every 30 minutes until resolution.
4. Conduct Post-Mortem (Blameless Incident Review) within 48 hours.

## 4. Escalation Matrix
| Severity | First Response | Escalation (15 min) | Management Notify |
|----------|---------------|---------------------|-------------------|
| P1       | On-Call Eng    | Secondary + TL       | VP Engineering    |
| P2       | On-Call Eng    | Team Lead            | Eng Manager       |
| P3       | On-Call Eng    | —                    | —                 |
| P4       | Triage queue   | —                    | —                 |

## 5. Post-Incident Review
Every P1 and P2 incident must have a blameless post-mortem document within 48 hours. The document must include: timeline, root cause, impact assessment, and at least 2 actionable remediation items with assigned owners and deadlines.
""",
)

MD_DATA_GOVERNANCE = (
    "data_governance_policy.md",
    """# Acme Corp Data Governance & Classification Policy

## 1. Data Classification Levels

### 1.1 Confidential
Data that could cause significant financial or reputational harm if disclosed. Examples: customer PII, financial reports, trade secrets, source code for proprietary algorithms.

### 1.2 Internal
Data intended for internal use only. Examples: internal memos, project plans, employee directories, meeting notes.

### 1.3 Public
Data explicitly approved for external sharing. Examples: press releases, marketing materials, public API documentation.

## 2. Data Handling Requirements

| Classification | Storage Encryption | Transit Encryption | Access Control | Retention |
|---------------|-------------------|-------------------|---------------|-----------|
| Confidential  | AES-256 at rest    | TLS 1.3 mandatory | RBAC + MFA     | 7 years   |
| Internal      | AES-256 at rest    | TLS 1.2+ required | RBAC           | 3 years   |
| Public        | Optional           | TLS recommended    | Open           | Indefinite|

## 3. Data Retention & Deletion
All classified data must follow the retention schedule. Upon expiration, data must be securely deleted using NIST SP 800-88 Rev.1 compliant methods. Deletion must be logged and auditable.

## 4. Cross-Border Data Transfer
Any transfer of Confidential or Internal data outside the company's primary jurisdiction requires Data Protection Officer (DPO) approval and must comply with applicable regulations (GDPR, CCPA, HIPAA as applicable).

## 5. Incident Reporting
Any suspected data breach must be reported to the Security Operations Center (SOC) within 1 hour of discovery via the #security-incidents Slack channel or security@acmecorp.com.
""",
)

# ---------------------------------------------------------------------------
# Plain Text Documents
# ---------------------------------------------------------------------------

TXT_SECURITY_POLICY = (
    "it_security_policy.txt",
    """Acme Corp IT Security Guidelines & Password Management

Password Policy:
- Minimum length: 16 characters.
- Must contain uppercase, lowercase, numbers, and special symbols.
- Passwords must be updated every 90 days.
- Re-use of any of the last 10 passwords is strictly prohibited.
- Passwords must not contain the user's name, email, or common dictionary words.

Multi-Factor Authentication (MFA):
- MFA is mandatory for all internal tools, VPNs, and email access.
- Only hardware tokens (YubiKey) or authenticator apps (1Password/Okta) are permitted. SMS MFA is disallowed due to SIM-swapping risks.
- MFA enrollment must be completed within 48 hours of account provisioning.

Remote Access & VPN:
- All company laptops must run the WireGuard corporate VPN when connected to public Wi-Fi networks.
- Split-tunneling is disabled by default. Exceptions require CISO approval.
- VPN sessions are limited to 12 hours and require re-authentication.

Device Security:
- Full-disk encryption (BitLocker/FileVault) must be enabled on all company devices.
- Automatic screen lock must be configured to activate after 5 minutes of inactivity.
- Personal devices are prohibited from accessing Confidential data unless enrolled in the MDM program.
""",
)

TXT_EXPENSE_POLICY = (
    "expense_reimbursement_policy.txt",
    """Acme Corp Employee Expense Reimbursement Policy
Last Updated: January 2026

1. ELIGIBLE EXPENSES
   - Business travel (flights, hotels, ground transportation)
   - Client entertainment meals (limit: $150 per person per event)
   - Conference and training registration fees (pre-approved by manager)
   - Home office equipment (one-time allowance of $1,500 for new hires)
   - Software subscriptions required for role (pre-approved by IT)

2. EXPENSE LIMITS
   - Domestic airfare: Economy class for flights under 6 hours
   - International airfare: Business class permitted for flights over 8 hours
   - Hotel: Up to $250/night in Tier 1 cities, $175/night elsewhere
   - Meals while traveling: $75/day per diem (receipts required for amounts over $25)
   - Ride-sharing: Uber/Lyft permitted; no surge pricing over 2.0x without approval

3. SUBMISSION PROCESS
   - All expense reports must be submitted within 30 days of incurring the expense.
   - Receipts are required for any single expense exceeding $25.
   - Submit via the Concur expense management system.
   - Manager approval required within 5 business days.
   - Finance processes approved expenses in the next bi-weekly payroll cycle.

4. NON-REIMBURSABLE EXPENSES
   - Personal entertainment, alcohol (unless client-facing), gym memberships,
     airline lounge memberships, first-class upgrades, personal phone bills,
     traffic violations, and any expense without a valid receipt.
""",
)

# ---------------------------------------------------------------------------
# Tabular Documents (CSV, TSV, JSONL)
# ---------------------------------------------------------------------------

CSV_BUDGET = (
    "department_budget_2026.csv",
    """Department,Budget_USD,Lead,Quarter,Headcount,Cost_Center
Engineering,4500000,Alex Rivers,Q1,120,CC-1001
Marketing,1200000,Sarah Jenkins,Q1,35,CC-2001
Human Resources,600000,David Miller,Q1,18,CC-3001
Product Design,900000,Elena Rostova,Q1,22,CC-4001
Sales,2100000,Marcus Chen,Q1,65,CC-5001
Customer Support,800000,Lisa Park,Q1,45,CC-6001
Legal & Compliance,500000,Robert Kim,Q1,12,CC-7001
Finance,450000,Jennifer Wu,Q1,15,CC-8001
""",
)

TSV_EMPLOYEE_DIR = (
    "employee_directory.tsv",
    """EmployeeID\tName\tDepartment\tTitle\tEmail\tOffice\tStart_Date
E001\tAlex Rivers\tEngineering\tVP Engineering\talex.rivers@acmecorp.com\tSF-HQ\t2019-03-15
E002\tSarah Jenkins\tMarketing\tVP Marketing\tsarah.jenkins@acmecorp.com\tSF-HQ\t2020-01-10
E003\tDavid Miller\tHuman Resources\tHR Director\tdavid.miller@acmecorp.com\tNY-Office\t2018-07-22
E004\tElena Rostova\tProduct Design\tHead of Design\telena.rostova@acmecorp.com\tSF-HQ\t2021-05-03
E005\tMarcus Chen\tSales\tVP Sales\tmarcus.chen@acmecorp.com\tNY-Office\t2020-11-01
E006\tLisa Park\tCustomer Support\tSupport Director\tlisa.park@acmecorp.com\tAustin\t2022-02-14
E007\tRobert Kim\tLegal\tGeneral Counsel\trobert.kim@acmecorp.com\tSF-HQ\t2019-09-30
E008\tJennifer Wu\tFinance\tCFO\tjennifer.wu@acmecorp.com\tSF-HQ\t2017-04-18
""",
)

JSONL_PRODUCT_CATALOG = (
    "product_catalog.jsonl",
    """{"product_id": "PROD-001", "name": "Enterprise Analytics Suite", "category": "Software", "price_usd": 15000, "license_type": "annual", "description": "Full-featured business intelligence platform with real-time dashboards, scheduled reports, and data warehouse integration."}
{"product_id": "PROD-002", "name": "Cloud Infrastructure Manager", "category": "DevOps", "price_usd": 8500, "license_type": "annual", "description": "Multi-cloud infrastructure provisioning and monitoring tool supporting AWS, Azure, and GCP with Terraform integration."}
{"product_id": "PROD-003", "name": "Customer 360 Platform", "category": "CRM", "price_usd": 12000, "license_type": "annual", "description": "Unified customer data platform combining CRM, support tickets, and engagement analytics into a single view."}
{"product_id": "PROD-004", "name": "SecureVault", "category": "Security", "price_usd": 6000, "license_type": "annual", "description": "Enterprise secrets management solution with HSM backing, RBAC, audit logging, and automated key rotation."}
""",
)

# ---------------------------------------------------------------------------
# Source Code Files
# ---------------------------------------------------------------------------

PY_DATA_PROCESSOR = (
    "data_processor.py",
    '''"""
Data Processing Pipeline — Acme Corp ETL module.

This module implements the core data transformation pipeline used by the
Analytics team for daily batch processing of customer event data.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    """Represents a single customer interaction event."""

    event_id: str
    customer_id: str
    event_type: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


class DataProcessor:
    """Main ETL processor for customer event streams.

    Attributes:
        batch_size: Number of records per processing batch.
        max_retries: Maximum retry attempts for failed records.
    """

    def __init__(self, batch_size: int = 1000, max_retries: int = 3):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._processed_count = 0
        self._error_count = 0

    def process_batch(self, events: List[EventRecord]) -> List[EventRecord]:
        """Process a batch of events through the transformation pipeline.

        Steps:
            1. Validate schema
            2. Normalize timestamps to UTC
            3. Enrich with customer metadata
            4. Deduplicate by event_id
        """
        validated = [e for e in events if self._validate(e)]
        normalized = [self._normalize_timestamp(e) for e in validated]
        deduplicated = self._deduplicate(normalized)

        self._processed_count += len(deduplicated)
        logger.info(
            "Processed batch: %d events (%d valid, %d deduplicated)",
            len(events), len(validated), len(deduplicated),
        )
        return deduplicated

    def _validate(self, event: EventRecord) -> bool:
        """Validate event has required fields."""
        if not event.event_id or not event.customer_id:
            self._error_count += 1
            return False
        return True

    def _normalize_timestamp(self, event: EventRecord) -> EventRecord:
        """Normalize timestamp to ISO 8601 UTC format."""
        event.processed = True
        return event

    def _deduplicate(self, events: List[EventRecord]) -> List[EventRecord]:
        """Remove duplicate events by event_id."""
        seen = set()
        unique = []
        for event in events:
            if event.event_id not in seen:
                seen.add(event.event_id)
                unique.append(event)
        return unique

    @property
    def stats(self) -> Dict[str, int]:
        """Return processing statistics."""
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
        }
''',
)

SQL_SCHEMA = (
    "database_schema.sql",
    """-- Acme Corp Core Database Schema
-- PostgreSQL 15+ compatible
-- Last updated: 2026-01-15

-- =============================================================================
-- Customer Management Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    full_name       VARCHAR(200) NOT NULL,
    company_name    VARCHAR(200),
    plan_tier       VARCHAR(50) NOT NULL DEFAULT 'free'
                    CHECK (plan_tier IN ('free', 'starter', 'professional', 'enterprise')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_plan ON customers(plan_tier);

-- =============================================================================
-- Subscription & Billing Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    plan_tier       VARCHAR(50) NOT NULL,
    monthly_price   DECIMAL(10, 2) NOT NULL,
    billing_cycle   VARCHAR(20) NOT NULL DEFAULT 'monthly'
                    CHECK (billing_cycle IN ('monthly', 'annual')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'cancelled', 'expired', 'suspended'))
);

CREATE INDEX idx_subs_customer ON subscriptions(customer_id);
CREATE INDEX idx_subs_status ON subscriptions(status);

-- =============================================================================
-- Usage Analytics Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS api_usage (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     UUID NOT NULL REFERENCES customers(customer_id),
    endpoint        VARCHAR(500) NOT NULL,
    method          VARCHAR(10) NOT NULL,
    status_code     INTEGER NOT NULL,
    response_ms     INTEGER NOT NULL,
    tokens_used     INTEGER DEFAULT 0,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_customer ON api_usage(customer_id);
CREATE INDEX idx_usage_recorded ON api_usage(recorded_at);

-- Monthly usage aggregation view
CREATE OR REPLACE VIEW monthly_usage_summary AS
SELECT
    customer_id,
    DATE_TRUNC('month', recorded_at) AS month,
    COUNT(*) AS total_requests,
    AVG(response_ms) AS avg_response_ms,
    SUM(tokens_used) AS total_tokens,
    COUNT(*) FILTER (WHERE status_code >= 500) AS error_count
FROM api_usage
GROUP BY customer_id, DATE_TRUNC('month', recorded_at);
""",
)

SH_DEPLOY = (
    "deploy.sh",
    """#!/usr/bin/env bash
# =============================================================================
# Acme Corp Production Deployment Script
# Usage: ./deploy.sh [staging|production] [--skip-tests] [--dry-run]
# =============================================================================

set -euo pipefail

ENVIRONMENT="${1:-staging}"
SKIP_TESTS="${2:-}"
DRY_RUN="${3:-}"
DOCKER_REGISTRY="registry.acmecorp.com"
APP_NAME="acme-platform"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_SHA=$(git rev-parse --short HEAD)
IMAGE_TAG="${ENVIRONMENT}-${GIT_SHA}-${TIMESTAMP}"

echo "========================================"
echo "  Deploying ${APP_NAME}"
echo "  Environment: ${ENVIRONMENT}"
echo "  Image Tag:   ${IMAGE_TAG}"
echo "========================================"

# Validate environment
if [[ "${ENVIRONMENT}" != "staging" && "${ENVIRONMENT}" != "production" ]]; then
    echo "ERROR: Invalid environment '${ENVIRONMENT}'. Must be 'staging' or 'production'."
    exit 1
fi

# Run tests unless skipped
if [[ "${SKIP_TESTS}" != "--skip-tests" ]]; then
    echo "Running test suite..."
    python -m pytest tests/ -v --tb=short
    echo "All tests passed."
fi

# Build Docker image
echo "Building Docker image: ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}"
if [[ "${DRY_RUN}" != "--dry-run" ]]; then
    docker build -t "${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}" .
    docker push "${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}"
fi

# Deploy via kubectl
echo "Deploying to Kubernetes cluster (${ENVIRONMENT})..."
if [[ "${DRY_RUN}" != "--dry-run" ]]; then
    kubectl set image deployment/${APP_NAME} \\
        app="${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}" \\
        -n "${ENVIRONMENT}" \\
        --record

    # Wait for rollout
    kubectl rollout status deployment/${APP_NAME} -n "${ENVIRONMENT}" --timeout=300s
fi

echo "Deployment complete: ${IMAGE_TAG}"
""",
)

# ---------------------------------------------------------------------------
# Configuration Files
# ---------------------------------------------------------------------------

JSON_APP_CONFIG = (
    "application_config.json",
    """{
  "application": {
    "name": "Acme Platform",
    "version": "3.2.1",
    "environment": "production"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4,
    "timeout_seconds": 30,
    "max_request_size_mb": 50
  },
  "database": {
    "primary": {
      "host": "db-primary.internal.acmecorp.com",
      "port": 5432,
      "name": "acme_production",
      "pool_size": 20,
      "max_overflow": 10,
      "ssl_mode": "verify-full"
    },
    "read_replicas": [
      {"host": "db-replica-1.internal.acmecorp.com", "port": 5432},
      {"host": "db-replica-2.internal.acmecorp.com", "port": 5432}
    ]
  },
  "cache": {
    "provider": "redis",
    "host": "redis.internal.acmecorp.com",
    "port": 6379,
    "ttl_seconds": 3600,
    "max_connections": 50
  },
  "feature_flags": {
    "new_dashboard_ui": true,
    "ai_assistant_beta": false,
    "advanced_analytics": true,
    "sso_enforcement": true
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 600,
    "burst_size": 100
  }
}
""",
)

YAML_INFRA_CONFIG = (
    "infrastructure.yaml",
    """# Acme Corp Infrastructure Configuration
# Managed by Platform Engineering team

cluster:
  name: acme-prod-us-west-2
  provider: aws
  region: us-west-2
  kubernetes_version: "1.29"

node_pools:
  - name: general
    instance_type: m6i.2xlarge
    min_nodes: 3
    max_nodes: 12
    disk_size_gb: 100
    labels:
      workload-type: general

  - name: ml-inference
    instance_type: g5.2xlarge
    min_nodes: 1
    max_nodes: 4
    disk_size_gb: 200
    gpu_count: 1
    labels:
      workload-type: ml-inference

  - name: data-pipeline
    instance_type: r6i.4xlarge
    min_nodes: 2
    max_nodes: 8
    disk_size_gb: 500
    labels:
      workload-type: data-intensive

monitoring:
  prometheus:
    retention_days: 30
    scrape_interval: 15s
  grafana:
    admin_email: platform-team@acmecorp.com
  alerting:
    pagerduty_service_key: "${PAGERDUTY_KEY}"
    slack_webhook: "${SLACK_ALERTS_WEBHOOK}"
    escalation_policy:
      - severity: critical
        notify:
          - channel: pagerduty
          - channel: slack
            target: "#incidents"
      - severity: warning
        notify:
          - channel: slack
            target: "#platform-alerts"

backup:
  schedule: "0 2 * * *"
  retention_days: 90
  storage:
    bucket: acme-backups-us-west-2
    encryption: AES-256
""",
)

HTML_API_DOCS = (
    "api_reference.html",
    """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Acme Platform API Reference</title>
</head>
<body>
    <h1>Acme Platform REST API Reference</h1>

    <h2>Authentication</h2>
    <p>All API requests require a Bearer token in the Authorization header.</p>
    <pre>Authorization: Bearer &lt;your-api-key&gt;</pre>

    <h2>Endpoints</h2>

    <h3>GET /api/v2/customers</h3>
    <p>List all customers with optional filtering and pagination.</p>
    <p><strong>Query Parameters:</strong></p>
    <ul>
        <li><code>page</code> (integer, default: 1) — Page number</li>
        <li><code>per_page</code> (integer, default: 25, max: 100) — Results per page</li>
        <li><code>plan_tier</code> (string, optional) — Filter by plan: free, starter, professional, enterprise</li>
        <li><code>is_active</code> (boolean, optional) — Filter by active status</li>
    </ul>

    <h3>POST /api/v2/customers</h3>
    <p>Create a new customer account.</p>
    <p><strong>Request Body (JSON):</strong></p>
    <pre>{
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "company_name": "Acme Inc",
  "plan_tier": "starter"
}</pre>

    <h3>GET /api/v2/usage/summary</h3>
    <p>Retrieve usage summary for the authenticated customer.</p>
    <p><strong>Response:</strong></p>
    <pre>{
  "total_requests": 15420,
  "total_tokens": 2340000,
  "avg_response_ms": 145,
  "billing_period": "2026-01"
}</pre>

    <h2>Rate Limiting</h2>
    <p>API requests are rate-limited to 600 requests per minute per API key.
       Rate limit headers are included in every response:</p>
    <ul>
        <li><code>X-RateLimit-Limit</code>: Maximum requests per minute</li>
        <li><code>X-RateLimit-Remaining</code>: Remaining requests in current window</li>
        <li><code>X-RateLimit-Reset</code>: Unix timestamp when the window resets</li>
    </ul>
</body>
</html>
""",
)


def generate_sample_docs(output_dir: str = SAMPLE_DOCS_DIR) -> None:
    """Generate all sample documents to the specified directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    all_docs = [
        # Markdown
        MD_VACATION_POLICY,
        MD_ONCALL_SOP,
        MD_DATA_GOVERNANCE,
        # Plain Text
        TXT_SECURITY_POLICY,
        TXT_EXPENSE_POLICY,
        # Tabular
        CSV_BUDGET,
        TSV_EMPLOYEE_DIR,
        JSONL_PRODUCT_CATALOG,
        # Source Code
        PY_DATA_PROCESSOR,
        SQL_SCHEMA,
        SH_DEPLOY,
        # Config / Web
        JSON_APP_CONFIG,
        YAML_INFRA_CONFIG,
        HTML_API_DOCS,
    ]

    for filename, content in all_docs:
        filepath = path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✔ Generated: {filepath.resolve()}")

    print(f"\n  ✅ {len(all_docs)} sample documents generated in '{output_dir}'")
    print(f"  File types: {', '.join(sorted({Path(f).suffix for f, _ in all_docs}))}")


if __name__ == "__main__":
    generate_sample_docs()
