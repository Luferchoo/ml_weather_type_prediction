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

# Definir pipeline


# Pipeline que ejecuta ambos modelos en paralelo
@pipeline(
    compute="cpu-cluster",
    description="Weather pipeline: ambos modelos"
)
def weather_pipeline_both(raw: Input,
                        subscription_id: str,
                        resource_group_name: str,
                        workspace_name: str):
    s = select_cols(raw_data=raw)
    imp = impute(data=s.outputs.selected)
    enc = encode(data=imp.outputs.imputed)
    sc = scale(data=enc.outputs.encoded)
    sp = split(data=sc.outputs.scaled)

    # Logistic Regression
    tr_lr = train_lr_unique_20250917(train_data=sp.outputs.train_data, register_model=None)
    sc_lr = score(test_data=sp.outputs.test_data, model=tr_lr.outputs.model_out)
    ev_lr = evalc(test_data=sp.outputs.test_data, predictions=sc_lr.outputs.predictions)
    pub_lr = publish_endpoint(
        model_input_path=tr_lr.outputs.model_out,
        endpoint_name="weather-lr-endpoint",
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )

    # Random Forest
    tr_rf = train_rf(train_data=sp.outputs.train_data)
    sc_rf = score(test_data=sp.outputs.test_data, model=tr_rf.outputs.model_out)
    ev_rf = evalc(test_data=sp.outputs.test_data, predictions=sc_rf.outputs.predictions)
    pub_rf = publish_endpoint(
        model_input_path=tr_rf.outputs.model_out,
        endpoint_name="weather-rf-endpoint",
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )

    return {
        "metrics_lr": ev_lr.outputs.metrics,
        "model_lr": tr_lr.outputs.model_out,
        "endpoint_lr": pub_lr.outputs.endpoint,
        "metrics_rf": ev_rf.outputs.metrics,
        "model_rf": tr_rf.outputs.model_out,
        "endpoint_rf": pub_rf.outputs.endpoint
    }


# Ejecuta ambos modelos:
pipeline_job = weather_pipeline_both(
    raw=Input(type="uri_file", path="azureml:weather_dataset:1"),
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    workspace_name=ml_client.workspace_name
)

# Enviar el pipeline
pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"🚀 Pipeline enviado: {pipeline_job.name}")
print(f"🔗 Link al portal: {pipeline_job.studio_url}")
