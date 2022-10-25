#!/bin/bash
cd "$SCRATCH" || exit 1

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
mkdir -p aws-cli/aws-cli
./aws/install --bin-dir aws-cli --install-dir aws-cli/aws-cli
AWS=$(find "$SCRATCH"/aws-cli -path "*bin/aws")
echo "Installation successful. Binary located at $AWS."
echo "You should do \`export AWS=$AWS\` before running download scripts."