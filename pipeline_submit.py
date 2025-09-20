from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, Output
from azure.ai.ml.entities import PipelineJob, Data
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# Conexión al workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b8acb497-503b-459f-bd2a-2fbee23b5f9e",
    resource_group_name="rg-mlops-ucb",
    workspace_name="ws-mlops-ucb2"
)
'''
# Crear el Data Asset
data_asset = Data(
    path="data/weather_classification_data.csv",  
    type="uri_file",  
    name="weather_dataset",
    description="Dataset sintético para clasificación de tipos de clima",
    version="1",
    tags={"tema": "clima", "tipo": "sintético", "uso": "preprocesamiento"}
)

# Registrar el asset en el workspace
ml_client.data.create_or_update(data_asset)
'''
# Cargar componentes desde YAML
select_cols = load_component(source="./components/select_cols/select_cols.yml")
impute = load_component(source="./components/impute/impute.yml")
encode = load_component(source="./components/encode/encode.yml")
scale = load_component(source="./components/scale/scale.yml")
split = load_component(source="./components/split/split.yml")

train_lr_unique_20250917 = load_component(source="./components/train_lr/train_lr.yml")
train_rf = load_component(source="./components/train_rf/train_rf.yml")
score = load_component(source="./components/score/score.yml")
evalc = load_component(source="./components/eval/eval.yml")
publish_endpoint = load_component(source="./components/publish_endpoint/publish_endpoint.yml")
language_service_sentiment = load_component(source="./components/language_service/language_service.yml")
compare_models = load_component(source="./components/compare_models/compare_models.yml")
select_best_model = load_component(source="./components/select_best_model/select_best_model.yml")

# Definir pipeline


# Pipeline que ejecuta ambos modelos en paralelo
@pipeline(
    compute="cpu-cluster",
    description="Weather pipeline: ambos modelos"
)
def weather_pipeline_both(raw: Input,
                        subscription_id: str,
                        resource_group_name: str,
                        workspace_name: str,
                        language_service_endpoint: str,
                        language_service_key: str): # Vuelvo a añadir estos parámetros
    s = select_cols(raw_data=raw)
    imp = impute(data=s.outputs.selected)
    enc = encode(data=imp.outputs.imputed)
    sc = scale(data=enc.outputs.encoded)
    
    # Nuevo nodo para el Servicio de Lenguaje
    ls_sentiment = language_service_sentiment(
        input_data=sc.outputs.scaled,
        text_column_name="Weather Type",  # Asegurando que se pase explícitamente
        language_service_endpoint=language_service_endpoint, # Se pasa como argumento normal
        language_service_key=language_service_key # Se pasa como argumento normal
    )

    sp = split(data=ls_sentiment.outputs.output_data)

    # Logistic Regression
    tr_lr = train_lr_unique_20250917(train_data=sp.outputs.train_data, register_model=None)
    sc_lr = score(test_data=sp.outputs.test_data, model=tr_lr.outputs.model_out)
    # ev_lr = evalc(test_data=sp.outputs.test_data, predictions=sc_lr.outputs.predictions) # Eliminado
    
    # Random Forest
    tr_rf = train_rf(train_data=sp.outputs.train_data)
    sc_rf = score(test_data=sp.outputs.test_data, model=tr_rf.outputs.model_out)
    # ev_rf = evalc(test_data=sp.outputs.test_data, predictions=sc_rf.outputs.predictions) # Eliminado

    # Nuevo nodo para comparar modelos
    compare_models_node = compare_models(
        test_data=sp.outputs.test_data,
        lr_predictions=sc_lr.outputs.predictions,
        rf_predictions=sc_rf.outputs.predictions
    )

    # Nuevo nodo para seleccionar el mejor modelo
    select_best_model_node = select_best_model(
        best_model_name_in=compare_models_node.outputs.best_model_name_out,
        lr_model_path=tr_lr.outputs.model_out,
        rf_model_path=tr_rf.outputs.model_out
    )

    # Publicar el mejor modelo
    pub_best_model = publish_endpoint(
        model_input_path=select_best_model_node.outputs.selected_model_output,
        endpoint_name="weather-best-model-endpoint", # Nuevo nombre para el endpoint
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )

    return {
        "model_lr": tr_lr.outputs.model_out,
        "model_rf": tr_rf.outputs.model_out,
        "compare_metrics": compare_models_node.outputs.output_metrics,
        "compare_report": compare_models_node.outputs.output_report,
        "selected_model": select_best_model_node.outputs.selected_model_output,
        "best_model_endpoint": pub_best_model.outputs.endpoint
    }


# Ejecuta ambos modelos:
pipeline_job = weather_pipeline_both(
    raw=Input(type="uri_file", path="azureml:weather_dataset:1"),
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    workspace_name=ml_client.workspace_name,
    language_service_endpoint="https://weather-pipe.cognitiveservices.azure.com/", # Tu endpoint real
    language_service_key="3Kc5R6BuEwRVcSQLY3aDhDAQU0cwgA8lky7rAk01KGIOW3ZF6jVmJQQJ99BIACLArgHXJ3w3AAAaACOGEt7v" # ¡Reemplaza esto con tu clave real!
)

# Enviar el pipeline
pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"🚀 Pipeline enviado: {pipeline_job.name}")
print(f"🔗 Link al portal: {pipeline_job.studio_url}")
