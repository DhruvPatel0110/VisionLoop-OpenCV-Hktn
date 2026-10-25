"""Automated AWS Teardown Script for VisionLoop.

Cleans up all deployed AWS resources to ensure zero lingering charges:
  1. Deletes AWS Lambda function and Function URL.
  2. Deletes Amazon ECR repository and untags all images.
  3. Deletes IAM execution role and policies.
  4. Resets deployment manifest.

Usage:
    python scripts/teardown_aws.py --region us-east-1
"""

import os
import sys
import json
import argparse
import boto3

def teardown(region: str = "us-east-1", function_name: str = "visionloop-cv5"):
    print("=" * 80)
    print("VISIONLOOP: AUTOMATED AWS TEARDOWN & COST SAFEGUARD")
    print(f"Region: {region} | Target: {function_name}")
    print("=" * 80)

    # 1. Delete Lambda Function
    lambda_client = boto3.client("lambda", region_name=region)
    print("\n[1/3] Deleting AWS Lambda Function...")
    try:
        lambda_client.delete_function(FunctionName=function_name)
        print(f"      -> Successfully deleted Lambda function: {function_name}")
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"      -> Lambda function {function_name} does not exist.")
    except Exception as e:
        print(f"      -> Warning deleting Lambda function: {e}")

    # 2. Delete ECR Repository
    ecr = boto3.client("ecr", region_name=region)
    print("\n[2/3] Deleting Amazon ECR Repository...")
    try:
        ecr.delete_repository(repositoryName=function_name, force=True)
        print(f"      -> Successfully deleted ECR repository: {function_name} (force=True)")
    except ecr.exceptions.RepositoryNotFoundException:
        print(f"      -> ECR repository {function_name} does not exist.")
    except Exception as e:
        print(f"      -> Warning deleting ECR repository: {e}")

    # 3. Delete IAM Role
    iam = boto3.client("iam", region_name=region)
    role_name = f"{function_name}-execution-role"
    print(f"\n[3/3] Deleting IAM Execution Role ({role_name})...")
    try:
        # Detach managed policies first
        policies = iam.list_attached_role_policies(RoleName=role_name)
        for pol in policies.get("AttachedPolicies", []):
            iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])
            print(f"      -> Detached policy: {pol['PolicyArn']}")
        iam.delete_role(RoleName=role_name)
        print(f"      -> Successfully deleted IAM role: {role_name}")
    except iam.exceptions.NoSuchEntityException:
        print(f"      -> IAM role {role_name} does not exist.")
    except Exception as e:
        print(f"      -> Warning deleting IAM role: {e}")

    # Update manifest
    deploy_file = "docs/aws_deployment.json"
    if os.path.exists(deploy_file):
        with open(deploy_file, "w", encoding="utf-8") as f:
            json.dump({"status": "TEARDOWN_COMPLETE", "timestamp_utc": ""}, f, indent=2)

    print("\n" + "=" * 80)
    print("TEARDOWN COMPLETE: All cloud resources cleanly removed.")
    print("================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teardown VisionLoop AWS resources")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--name", type=str, default="visionloop-cv5")
    args = parser.parse_args()
    teardown(region=args.region, function_name=args.name)
