#!/bin/bash
DATA="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$DATA" || exit 1

$AWS --version || echo "Something went wrong. Make sure you have aws-cli installed and the path to the binary saved in \$AWS. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"

mkdir -p Park_v_Control
cd Park_v_Control || exit 1
$AWS s3 sync --no-sign-request s3://openneuro.org/ds001907 ds001907-download/

mkdir -p Rest_v_LearningRecall
cd Rest_v_LearningRecall || exit 1
$AWS s3 sync --no-sign-request s3://openneuro.org/ds001454 ds001454-download/

mkdir -p Rest_w_Depression_v_Control
cd Rest_w_Depression_v_Control || exit 1
$AWS s3 sync --no-sign-request s3://openneuro.org/ds002748 ds002748-download/

mkdir -p Rest_w_Healthy_v_OsteoPain
cd Rest_w_Healthy_v_OsteoPain || exit 1
$AWS s3 sync --no-sign-request s3://openneuro.org/ds000208 ds000208-download/

mkdir -p Rest_w_VigilanceAttention
cd Rest_w_VigilanceAttention || exit 1
$AWS s3 sync --no-sign-request s3://openneuro.org/ds001168 ds001168-download/

