from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import(
	Model, Environment, CodeConfiguration,
	ManagedOnlineEndpoint, ManagedOnlineDeployment
)

import datetime, random, json

#1) conectar al wokspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b8acb497-503b-459f-bd2a-2fbee23b5f9e",
    resource_group_name="rg-mlops-ucb",
    workspace_name="ws-mlops-ucb2"
)

# 2) Registrar (o referenciar) el modelo
registered_model = ml_client.models.get(name="weather-rf", version="7")

#3 Definir entorno(usa conda.yml de /src)
env = Environment(
	name="env-sklearm-online",
	conda_file="envs/conda.yml",
	description="Entorno para inferencia sklearn"
)

#4 Crear endpoint(nombre unico)
suffix = datetime.datetime.now().strftime("%m%d%H%M")+ str(random.randint(10,99))
endpoint= ManagedOnlineEndpoint(
	name=f"weather-end-point-{suffix}",
	auth_mode="key",
	description="Endpoint clima"
)

ml_client.begin_create_or_update(endpoint).result()

# 5) Crear deployment (blue) con el model y el codigo de inferencia

deployment = ManagedOnlineDeployment(
	name="deploy",
	endpoint_name=endpoint.name,
	model=registered_model,
	environment= env,
	code_configuration=CodeConfiguration(
		code="components/score",
		scoring_script="score.py"
	),
	instance_type="Standard_DS3_v2",
	instance_count=1
)
ml_client.begin_create_or_update(deployment).result()

#6 Enrutar trafico al deployment
endpoint.traffic = {"blue":100}
ml_client.begin_create_or_update(endpoint).result()

print(f"Endpoint listo: {endpoint.name}")

keys = ml_client.online_endpoints.get_keys(name=endpoint.name)

print(f"clave primaria:{keys.primary_key}")
