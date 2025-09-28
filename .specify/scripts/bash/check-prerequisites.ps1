# PowerShell script to check prerequisites

# Check if running in a git repository
$gitCheck = git rev-parse --is-inside-work-tree 2>$null
if (-not $gitCheck) {
    Write-Error "Error: Not in a git repository"
    exit 1
}

# Get the feature branch name
$featureBranch = git rev-parse --abbrev-ref HEAD 2>$null
if (-not $featureBranch) { $featureBranch = "main" }

# Set default values
$featureDir = ".specify\features\$featureBranch"
$featureSpec = "$featureDir\spec.md"
$implPlan = "$featureDir\plan.md"
$tasks = "$featureDir\tasks.md"

# Output as JSON if requested
if ($args -contains "--json") {
    $output = @{
        FEATURE_DIR = $featureDir
        FEATURE_SPEC = $featureSpec
        IMPL_PLAN = $implPlan
        TASKS = $tasks
    } | ConvertTo-Json
    
    if ($args -contains "--paths-only") {
        $output
    } else {
        $output
    }
    exit 0
}

# Default output
Write-Output "Feature Directory: $featureDir"
Write-Output "Feature Specification: $featureSpec"
Write-Output "Implementation Plan: $implPlan"
Write-Output "Tasks: $tasks"
