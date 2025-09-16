from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl
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
train_lr = load_component(source="./components/train_lr/train_lr.yml")
score = load_component(source="./components/score/score.yml")
evalc = load_component(source="./components/eval/eval.yml")

# Definir pipeline
@pipeline(compute="cpu-cluster", description="Weather pipeline: 9 steps")
def weather_pipeline(raw: Input):
    s = select_cols(raw_data=raw)
    imp = impute(data=s.outputs.selected)
    enc = encode(data=imp.outputs.imputed)
    sc = scale(data=enc.outputs.encoded)
    sp = split(data=sc.outputs.scaled)
    tr = train_lr(train_data=sp.outputs.train_data)
    sc2 = score(test_data=sp.outputs.test_data, model=tr.outputs.model)
    ev = evalc(test_data=sp.outputs.test_data, predictions=sc2.outputs.predictions)
    return {
        "metrics": ev.outputs.metrics,
        "model": tr.outputs.model
    }

# Crear instancia del pipeline
pipeline_job = weather_pipeline(
    raw=Input(type="uri_file", path="azureml:weather_dataset:1")
)

# Enviar el pipeline
pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"🚀 Pipeline enviado: {pipeline_job.name}")
print(f"🔗 Link al portal: {pipeline_job.studio_url}")
