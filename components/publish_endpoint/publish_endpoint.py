import argparse
import os
import joblib
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import ManagedIdentityCredential
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="weather-prediction-endpoint")
    parser.add_argument("--endpoint_output_path", type=str, required=True)
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group_name", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    args = parser.parse_args()

    # Configuración de MLClient
    try:
        ml_client = MLClient(
            credential=ManagedIdentityCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group_name,
            workspace_name=args.workspace_name
        )
    except Exception as e:
        print(f"Error al inicializar MLClient con ManagedIdentityCredential: {e}")
        print("Asegúrese de que la Managed Identity tenga los permisos necesarios (por ejemplo, 'Colaborador' en el Grupo de Recursos y 'Colaborador de AzureML' en el Workspace) y que el entorno donde se ejecuta este código esté configurado para usar una Managed Identity.")
        raise

    print(f"Publicando modelo desde: {args.model_input_path}")
    print(f"Nombre del endpoint: {args.endpoint_name}")

    # Cargar el modelo para determinar la clase
    model_path = os.path.join(args.model_input_path, "model.pkl")
    
    try:
        model_artifact = joblib.load(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo desde {model_path}: {e}")
        raise

    model_type = "random_forest_model" if "RandomForestClassifier" in str(type(model_artifact)) else "logistic_regression_model"

    # Registrar el modelo
    registered_model_name = f"{args.endpoint_name}-{model_type}-model"
    
    try:
        model = ml_client.models.create_or_update(
            Model(
                path=args.model_input_path,
                name=registered_model_name,
                description="Modelo entrenado para predicción de tipo de clima",
                type="custom_model" # or "sklearn"
            )
        )
    except Exception as e:
        print(f"Error al registrar el modelo {registered_model_name}: {e}")
        raise

    print(f"Modelo registrado con nombre: {model.name}, versión: {model.version}")

    # Crear o actualizar el endpoint
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Endpoint para el modelo de predicción de tipo de clima",
        auth_mode="key"
    )
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    except Exception as e:
        print(f"Error al crear/actualizar el endpoint {endpoint.name}: {e}")
        raise
    print(f"Endpoint '{endpoint.name}' creado/actualizado.")

    # Crear deployment
    deployment_name = "blue"
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint.name,
        model=model.id,
        instance_type="Standard_DS2_v2",
        instance_count=1
    )
    try:
        ml_client.online_deployments.begin_create_or_update(deployment).wait()
    except Exception as e:
        print(f"Error al crear/actualizar el deployment {deployment.name} en el endpoint {endpoint.name}: {e}")
        raise
    print(f"Deployment '{deployment.name}' creado/actualizado en el endpoint '{endpoint.name}'.")

    # Establecer el deployment como predeterminado
    endpoint.traffic = {deployment_name: 100}
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    except Exception as e:
        print(f"Error al establecer el tráfico del endpoint {endpoint.name}: {e}")
        raise
    print(f"Tráfico del endpoint dirigido al deployment '{deployment_name}'.")

    # Escribir el nombre del endpoint en el archivo de salida
    os.makedirs(args.endpoint_output_path, exist_ok=True)
    with open(os.path.join(args.endpoint_output_path, "endpoint_name.txt"), "w") as f:
        f.write(endpoint.name)

if __name__ == "__main__":
    main()
