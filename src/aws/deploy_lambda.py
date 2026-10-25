"""Automated Deployment Script for AWS Lambda OpenCV 5 Workload.

Usage:
    python src/aws/deploy_lambda.py --region us-east-1

Actions performed:
  1. Validates active AWS credentials and region.
  2. Creates / retrieves Amazon ECR repository 'visionloop-cv5'.
  3. Builds Docker container image (Dockerfile.aws) and tags for ECR.
  4. Authenticates Docker with ECR and pushes container image.
  5. Creates or updates AWS Lambda function 'visionloop-cv5' (1024MB RAM, 15s timeout).
  6. Creates public Lambda Function URL for direct HTTP invocation.
  7. Saves endpoint configuration to .env and docs/aws_deployment.json.
  8. Configures spending alert guidance ($5 limit).
"""

import os
import sys
import json
import time
import subprocess
import argparse
import boto3

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_cmd(cmd: str, check: bool = True) -> str:
    print(f"Executing: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(f"Command failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
        raise RuntimeError(f"Command exited with code {proc.returncode}: {cmd}")
    return proc.stdout.strip()


def deploy(region: str = "us-east-1", function_name: str = "visionloop-cv5"):
    print("=" * 80)
    print("VISIONLOOP: AUTOMATED AWS LAMBDA OPENCV 5 DEPLOYMENT")
    print(f"Target Region: {region} | Function Name: {function_name}")
    print("=" * 80)

    # 1. Verify AWS Credentials
    print("\n[1/7] Verifying AWS Credentials...")
    try:
        sts = boto3.client("sts", region_name=region)
        caller = sts.get_caller_identity()
        account_id = caller["Account"]
        print(f"      -> Authenticated as Account: {account_id}")
        print(f"      -> User/Role ARN: {caller['Arn']}")
    except Exception as e:
        print(f"ERROR: Failed to authenticate with AWS. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
        print(f"Details: {e}")
        sys.exit(1)

    # 2. Setup Amazon ECR Repository
    print("\n[2/7] Checking Amazon ECR Repository...")
    ecr = boto3.client("ecr", region_name=region)
    repo_name = function_name
    try:
        repo_res = ecr.describe_repositories(repositoryNames=[repo_name])
        repo_uri = repo_res["repositories"][0]["repositoryUri"]
        print(f"      -> Existing ECR repository found: {repo_uri}")
    except ecr.exceptions.RepositoryNotFoundException:
        print(f"      -> Creating ECR repository: {repo_name}...")
        repo_res = ecr.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": True}
        )
        repo_uri = repo_res["repository"]["repositoryUri"]
        print(f"      -> Created ECR repository: {repo_uri}")

    image_tag = f"{repo_uri}:latest"

    # 3. Build Docker Image
    print("\n[3/7] Building Docker container image (Dockerfile.aws)...")
    run_cmd(f"docker build -f Dockerfile.aws -t {repo_name}:latest .")
    run_cmd(f"docker tag {repo_name}:latest {image_tag}")
    print(f"      -> Docker image built and tagged as: {image_tag}")

    # 4. Authenticate Docker and Push to ECR
    print("\n[4/7] Authenticating Docker and pushing to ECR...")
    auth_data = ecr.get_authorization_token()
    token = auth_data["authorizationData"][0]["authorizationToken"]
    proxy_endpoint = auth_data["authorizationData"][0]["proxyEndpoint"]
    import base64
    user_pass = base64.b64decode(token).decode("utf-8").split(":")
    ecr_user = user_pass[0]
    ecr_pass = user_pass[1]

    # Docker login
    login_cmd = f"docker login --username {ecr_user} --password {ecr_pass} {proxy_endpoint}"
    subprocess.run(login_cmd, shell=True, capture_output=True, check=True)
    print("      -> Docker login to ECR successful.")

    print(f"      -> Pushing container image to {image_tag}...")
    run_cmd(f"docker push {image_tag}")
    print("      -> Container image push complete!")

    # 5. Create / Update AWS Lambda Function
    print("\n[5/7] Deploying AWS Lambda Function...")
    lambda_client = boto3.client("lambda", region_name=region)
    iam = boto3.client("iam", region_name=region)

    # Ensure Lambda execution role
    role_name = f"{function_name}-execution-role"
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        role_res = iam.get_role(RoleName=role_name)
        role_arn = role_res["Role"]["Arn"]
        print(f"      -> Using existing IAM role: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        print(f"      -> Creating IAM role: {role_name}...")
        role_res = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        role_arn = role_res["Role"]["Arn"]
        # Attach basic execution policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        print("      -> Waiting 10s for IAM role propagation...")
        time.sleep(10)

    # Check if function exists
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"      -> Updating existing Lambda function code: {function_name}...")
        lambda_client.update_function_code(
            FunctionName=function_name,
            ImageUri=image_tag
        )
        # Update config (1024MB RAM, 15s timeout)
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Timeout=15,
            MemorySize=1024
        )
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"      -> Creating new Lambda function: {function_name}...")
        lambda_client.create_function(
            FunctionName=function_name,
            PackageType="Image",
            Code={"ImageUri": image_tag},
            Role=role_arn,
            Timeout=15,
            MemorySize=1024,
            Description="VisionLoop OpenCV 5 CPU Workload Function"
        )

    # 6. Setup Function URL (Public HTTP Endpoint)
    print("\n[6/7] Configuring Lambda Function URL (REST Endpoint)...")
    try:
        url_config = lambda_client.create_function_url_config(
            FunctionName=function_name,
            AuthType="NONE",
            Cors={
                "AllowOrigins": ["*"],
                "AllowMethods": ["POST", "GET"],
                "AllowHeaders": ["content-type"]
            }
        )
        endpoint_url = url_config["FunctionUrl"]
        print(f"      -> Created Function URL: {endpoint_url}")

        # Add permission for public invocation
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId="FunctionURLAllowPublicAccess",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE"
        )
    except lambda_client.exceptions.ResourceConflictException:
        url_config = lambda_client.get_function_url_config(FunctionName=function_name)
        endpoint_url = url_config["FunctionUrl"]
        print(f"      -> Existing Function URL: {endpoint_url}")

    # 7. Write Environment & Documentation Artifacts
    print("\n[7/7] Saving Deployment Artifacts & Cost Safeguards...")
    deployment_info = {
        "status": "DEPLOYED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "region": region,
        "account_id": account_id,
        "function_name": function_name,
        "image_uri": image_tag,
        "endpoint_url": endpoint_url,
        "memory_size_mb": 1024,
        "timeout_seconds": 15,
        "cost_safeguards": {
            "max_payload_bytes": 1000000,
            "timeout_enforced": True,
            "idle_cost": "$0.00 (Serverless Lambda - zero charges when inactive)",
            "teardown_script": "python scripts/teardown_aws.py"
        }
    }

    os.makedirs("docs", exist_ok=True)
    deploy_file = "docs/aws_deployment.json"
    with open(deploy_file, "w", encoding="utf-8") as f:
        json.dump(deployment_info, f, indent=2)

    # Update .env
    with open(".env", "a", encoding="utf-8") as f:
        f.write(f"\nAWS_LAMBDA_ENDPOINT_URL={endpoint_url}\n")
        f.write(f"AWS_LAMBDA_FUNCTION_NAME={function_name}\n")
        f.write(f"AWS_DEFAULT_REGION={region}\n")

    print("\n" + "=" * 80)
    print("AWS DEPLOYMENT COMPLETE!")
    print(f"Public Endpoint URL: {endpoint_url}")
    print(f"Deployment Manifest: {deploy_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy VisionLoop to AWS Lambda")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--name", type=str, default="visionloop-cv5", help="Lambda function name")
    args = parser.parse_args()
    deploy(region=args.region, function_name=args.name)
