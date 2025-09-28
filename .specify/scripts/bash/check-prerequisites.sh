#!/bin/bash

# Check if running in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get the feature branch name
FEATURE_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Set default values
FEATURE_DIR=".specify/features/${FEATURE_BRANCH}"
FEATURE_SPEC="${FEATURE_DIR}/spec.md"
IMPL_PLAN="${FEATURE_DIR}/plan.md"
TASKS="${FEATURE_DIR}/tasks.md"

# Create output JSON
if [[ "$1" == "--json" ]]; then
    jq -n --arg feature_dir "$FEATURE_DIR" \
           --arg feature_spec "$FEATURE_SPEC" \
           --arg impl_plan "$IMPL_PLAN" \
           --arg tasks "$TASKS" \
           '{
             "FEATURE_DIR": $feature_dir,
             "FEATURE_SPEC": $feature_spec,
             "IMPL_PLAN": $impl_plan,
             "TASKS": $tasks
           }'
    exit 0
fi

echo "Feature Directory: ${FEATURE_DIR}"
echo "Feature Specification: ${FEATURE_SPEC}"
echo "Implementation Plan: ${IMPL_PLAN}"
echo "Tasks: ${TASKS}"
