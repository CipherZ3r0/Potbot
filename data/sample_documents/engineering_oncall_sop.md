# Engineering On-Call Escalation SOP

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
