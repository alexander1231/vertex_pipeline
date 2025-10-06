from kfp import dsl, compiler

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform==1.66.0"]
)
def train_and_deploy_sdk(
    project_id: str,
    region: str,
    staging_bucket: str,
    bq_uri: str,
    target_column: str,
    model_display_name: str,
    endpoint_display_name: str,
    training_budget_hours: float = 1.0,
) -> str:
    from google.cloud import aiplatform
    import time, json

    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    ds = aiplatform.TabularDataset.create(
        display_name=f"ds-from-bq-{int(time.time())}",
        bq_source=bq_uri,
    )

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"automl-train-{int(time.time())}",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
    )
    model = job.run(
        dataset=ds,
        target_column=target_column,
        training_budget_milli_node_hours=int(training_budget_hours * 1000),
        model_display_name=model_display_name,
        disable_early_stopping=False,
        sync=True,
    )

    eps = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')
    endpoint = eps[0] if eps else aiplatform.Endpoint.create(display_name=endpoint_display_name, sync=True)

    deployed = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model_display_name}-deployed",
        traffic_percentage=100,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )

    return json.dumps({
        "dataset": ds.resource_name,
        "model": model.resource_name,
        "endpoint": endpoint.resource_name,
        "deployed_model_id": getattr(deployed, "id", None),
    })

@dsl.pipeline(name="vertex-automl-single-step-train-deploy")
def pipeline(
    project_id: str = "TU_PROYECTO",
    region: str = "us-central1",
    staging_bucket: str = "gs://TU_BUCKET",
    bq_uri: str = "bq://tu-proyecto.tu_dataset.tu_tabla",
    target_column: str = "label",
    model_display_name: str = "automl-model-demo",
    endpoint_display_name: str = "demo-endpoint",
    training_budget_hours: float = 1.0,
):
    _ = train_and_deploy_sdk(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
        bq_uri=bq_uri,
        target_column=target_column,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        training_budget_hours=training_budget_hours,
    )

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipeline.json")
    print("Compilado OK → pipeline.json")
