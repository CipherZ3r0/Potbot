"""
Sample Document Generator — Generates synthetic corporate policy and SOP files
(Markdown, Text, CSV) for peer-reviewer testing and reproducibility.
"""

import os
from pathlib import Path

SAMPLE_DOCS_DIR = "data/sample_documents"

DOC_1_TITLE = "company_vacation_policy.md"
DOC_1_CONTENT = """# Acme Corp Employee Vacation & Paid Time Off (PTO) Policy

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
"""

DOC_2_TITLE = "engineering_oncall_sop.md"
DOC_2_CONTENT = """# Engineering On-Call Escalation SOP

## 1. Incident Classification
- **P1 (Critical)**: Production service complete outage or data corruption. SLA resolution target: 1 hour.
- **P2 (High)**: Major feature broken impacting > 25% users. SLA resolution target: 4 hours.
- **P3 (Medium)**: Minor bug with available workaround. SLA resolution target: 24 hours.

## 2. On-Call Rotation Schedule
The primary on-call engineer shifts every Monday at 09:00 AM UTC. A secondary backup engineer is assigned to handle escalations if the primary fails to acknowledge a P1 alert within 15 minutes.

## 3. PagerDuty Alert Response
1. Acknowledge alert in PagerDuty within 15 minutes for P1 incidents.
2. Open a dedicated Slack incident channel (`#inc-YYYYMMDD-description`).
3. Post status update every 30 minutes until resolution.
4. Conduct Post-Mortem (Blameless Incident Review) within 48 hours.
"""

DOC_3_TITLE = "it_security_policy.txt"
DOC_3_CONTENT = """Acme Corp IT Security Guidelines & Password Management

Password Policy:
- Minimum length: 16 characters.
- Must contain uppercase, lowercase, numbers, and special symbols.
- Passwords must be updated every 90 days.
- Re-use of any of the last 10 passwords is strictly prohibited.

Multi-Factor Authentication (MFA):
- MFA is mandatory for all internal tools, VPNs, and email access.
- Only hardware tokens (YubiKey) or authenticator apps (1Password/Okta) are permitted. SMS MFA is disallowed due to SIM-swapping risks.

Remote Access & VPN:
- All company laptops must run the WireGuard corporate VPN when connected to public Wi-Fi networks.
"""

DOC_4_TITLE = "department_budget_2026.csv"
DOC_4_CONTENT = """Department,Budget_USD,Lead,Quarter
Engineering,4500000,Alex Rivers,Q1
Marketing,1200000,Sarah Jenkins,Q1
Human Resources,600000,David Miller,Q1
Product Design,900000,Elena Rostova,Q1
"""


def generate_sample_docs(output_dir: str = SAMPLE_DOCS_DIR) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    files = {
        DOC_1_TITLE: DOC_1_CONTENT,
        DOC_2_TITLE: DOC_2_CONTENT,
        DOC_3_TITLE: DOC_3_CONTENT,
        DOC_4_TITLE: DOC_4_CONTENT,
    }

    for filename, content in files.items():
        filepath = path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated sample doc: {filepath.resolve()}")

    print(f"\nAll sample documents generated successfully in '{output_dir}'.")


if __name__ == "__main__":
    generate_sample_docs()
