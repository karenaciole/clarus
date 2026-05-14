import os
from pathlib import Path
from aws_cdk import (
    App,
    Stack,
    aws_s3 as s3,
    RemovalPolicy,
    CfnOutput,
    Duration,
    aws_rds as rds,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env")

class InfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket_name = os.getenv("S3_BUCKET_NAME")
        db_name = os.getenv("DB_NAME", "clarus")
        db_username = os.getenv("DB_USER", "clarus_admin")
        ssh_key_name = os.getenv("SSH_KEY_NAME", "clarus-key")

        vpc = ec2.Vpc(self, "ClarusVPC",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="Isolated", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24)
            ]
        ) 

        vpc.add_gateway_endpoint("S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3)
        document_bucket = s3.Bucket(self, "ClarusDocumentBucket",
            bucket_name=bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        db = rds.DatabaseInstance(
            self,
            "ClarusDB",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_16),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO),
            database_name=db_name,
            credentials=rds.Credentials.from_generated_secret(db_username),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            removal_policy=RemovalPolicy.DESTROY,
            deletion_protection=False,
            allocated_storage=20,
            max_allocated_storage=30,
        )

        sg = ec2.SecurityGroup(self, "ClarusAppSG", vpc=vpc, allow_all_outbound=True)
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(8501), "Streamlit")
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(22), "SSH")
        db.connections.allow_default_port_from(sg)

        role = iam.Role(self, "ClarusRole", 
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly")
            ]
        )
        document_bucket.grant_read_write(role)
        if db.secret:
            db.secret.grant_read(role)
        
        role.add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:clarus/*"]
        ))
        
        ecr_image_uri = os.getenv("ECR_IMAGE_URI", "")
        config_secret_id = os.getenv("CONFIG_SECRET_ID", "clarus/prod/config")
        
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "dd if=/dev/zero of=/swapfile bs=128M count=16",
            "chmod 600 /swapfile",
            "mkswap /swapfile",
            "swapon /swapfile",
            "echo '/swapfile swap swap defaults 0 0' >> /etc/fstab",
            
            "dnf install -y docker",
            "systemctl start docker",
            "systemctl enable docker",
            "sleep 10", 
            
            f"aws ecr get-login-password --region {self.region} | docker login --username AWS --password-stdin {ecr_image_uri.split('/')[0]}",
            f"docker pull {ecr_image_uri}",
            f"docker run -d -p 8501:8501 --name clarus-app " +
            f"-e CONFIG_SECRET_ID='{config_secret_id}' " +
            f"-e AWS_REGION={self.region} " +
            f"{ecr_image_uri}"
        )

        instance = ec2.Instance(self, "ClarusInstance",
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(), 
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=sg,
            role=role,
            key_name=ssh_key_name,
            user_data=user_data, 
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(30, encrypted=True)
            )]
        )
        
        idle_alarm = cloudwatch.Alarm(self, "IdleStopAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/EC2",
                metric_name="CPUUtilization",
                dimensions_map={"InstanceId": instance.instance_id},
                period=Duration.minutes(5)
            ),
            threshold=5,
            evaluation_periods=6,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD
        )
        idle_alarm.add_alarm_action(cw_actions.Ec2Action(cw_actions.Ec2InstanceAction.STOP))
        
        CfnOutput(self, "EC2PublicIP", value=instance.instance_public_ip)
        CfnOutput(self, "RDSAddress", value=db.db_instance_endpoint_address)
        if db.secret:
            CfnOutput(self, "RDSSecretArn", value=db.secret.secret_arn)

app = App()
InfraStack(app, "ClarusInfraStack")
app.synth()
