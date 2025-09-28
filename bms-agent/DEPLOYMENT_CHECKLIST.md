# Deployment Checklist

This checklist guides the rollout of the BMS Agent to a RunPod pod. Complete each
section before moving to the next. Record results in your change log for auditability.

## 1. Pre-Deployment Validation
- **Environment sync**: Confirm `main` is merged into the deployment branch.
- **Tests**: Run `./scripts/run_tests.sh` locally and ensure the GitHub Actions
  pipeline (CI/CD Pipeline workflow) is green.
- **Secrets**: Double-check RunPod credentials and GitHub repository secrets:
  `BMS_API_KEY`, `DEPLOY_KEY`, optional `CODECOV_TOKEN`, `SAFETY_API_KEY`.
- **Artifacts**: Verify Ollama models (`snowflake-arctic-embed2`,
  `mistral-nemo:12b-instruct`) are pre-pulled on the pod or listed in rollout plan.
- **Backups**: Snapshot `~/persistent/qdrant_storage` and `~/persistent/bms_data`.

## 2. Deployment Steps
- **Sync code**:
  ```bash
  rsync -avz --delete ./bms-agent <user>@<runpod-ip>:~/bms-agent
  ```
- **Install requirements**:
  ```bash
  ssh <user>@<runpod-ip> 'cd ~/bms-agent && source venv/bin/activate && \
    pip install -r reqs/requirements.txt && pip install -r requirements-test.txt'
  ```
- **Apply configuration**:
  ```bash
  ssh <user>@<runpod-ip> 'echo "export BMS_API_KEY=<key>" >> ~/bms-agent/config/env.sh'
  ```
- **Restart services**:
  ```bash
  ssh <user>@<runpod-ip> '~/bms-agent/scripts/manage_services.sh restart'
  ```
- **Run smoke tests**:
  ```bash
  ssh <user>@<runpod-ip> '~/bms-agent/scripts/run_tests.sh'
  ```
- **Validate health**:
  ```bash
  ssh <user>@<runpod-ip> '~/bms-agent/scripts/health_check.sh'
  ```

## 3. Post-Deployment Verification
- **Functional check**: Use `/api/v1/search/semantic` with a known query and confirm
  relevant results.
- **Monitoring**: Tail `~/persistent/logs/api.log` and `~/persistent/logs/qdrant.log`
  for at least 10 minutes.
- **n8n/OpenWebUI**: Trigger Slack workflow and OpenWebUI tool to ensure
  integrations are intact.
- **Rollback plan**: If issues arise, restore backups and rerun
  `./scripts/manage_services.sh restart`.

## 4. Communication
- Notify stakeholders (engineering, product) with:
  - Deployment window and result
  - Version/commit SHA deployed
  - Known issues or follow-up tickets
- Update `tasks.md` and project tracker with deployment status and any new action
  items.

## 5. Continuous Improvement
- Capture lessons learned and feed them into `tasks.md` as new tasks.
- Consider automating manual steps in `.github/workflows/ci-cd.yml` deploy job.
- Review performance metrics (`reports/performance-baseline.md`) within 24 hours.
