import argparse
import pandas as pd
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def authenticate_client(endpoint, key):
    ta_credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return client

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--text_column_name", type=str, default="Weather Type") # Asumiendo una columna de texto
    parser.add_argument("--language_service_endpoint", type=str, required=True)
    parser.add_argument("--language_service_key", type=str, required=True)
    args = parser.parse_args()

    print(f"Iniciando el componente de Análisis de Sentimientos...")
    print(f"Leyendo datos desde: {args.input_data}")

    # Cargar los datos
    df = pd.read_csv(args.input_data)

    if args.text_column_name not in df.columns:
        print(f"ADVERTENCIA: La columna de texto '{args.text_column_name}' no se encuentra en los datos. Saltando el análisis de sentimientos.")
        df[f"{args.text_column_name}_sentiment_score"] = None
    else:
        # Autenticar cliente del Servicio de Lenguaje
        client = authenticate_client(args.language_service_endpoint, args.language_service_key)

        # Preparar documentos para el análisis de sentimientos
        documents = df[args.text_column_name].astype(str).tolist()

        all_sentiment_results = []
        # Dividir los documentos en lotes de 10 (límite del servicio)
        for i in range(0, len(documents), 10):
            batch = documents[i : i + 10]
            if not batch:  # Saltar lotes vacíos
                continue
            print(f"Realizando análisis de sentimientos en lote {i//10 + 1}...")
            try:
                batch_results = client.analyze_sentiment(documents=batch) # Realizar análisis de sentimientos por lote
                all_sentiment_results.extend(batch_results)
            except Exception as e:
                print(f"Error al procesar lote {i//10 + 1}: {e}")
                # En caso de error en un lote, añadir Nones para mantener la longitud
                for _ in batch:
                    all_sentiment_results.append({"is_error": True, "error": {"code": "BatchError", "message": str(e)}})

        # Extraer puntuaciones de sentimiento de todos los resultados
        sentiment_scores = []
        for doc_item in all_sentiment_results:
            # Verificar si es un objeto de resultado de Text Analytics o un diccionario de error
            if hasattr(doc_item, 'is_error') and doc_item.is_error:
                sentiment_scores.append(None)
                print(f"Error procesando documento: {getattr(doc_item, 'id', 'N/A')}, Error: {doc_item.error.code} - {doc_item.error.message}")
            elif isinstance(doc_item, dict) and doc_item.get("is_error"):
                sentiment_scores.append(None)
                error_info = doc_item.get("error", {})
                print(f'Error procesando documento (lote): {error_info.get("code", "N/A")} - {error_info.get("message", "Desconocido")}')
            elif hasattr(doc_item, 'sentiment') and hasattr(doc_item.sentiment, 'value'):
                sentiment_scores.append(doc_item.sentiment.value) # Positivo, Negativo, Neutral
            else:
                sentiment_scores.append(None)
                print(f"ADVERTENCIA: Resultado de sentimiento inesperado: {doc_item}. Se asignará None.")

        # Añadir las puntuaciones al DataFrame
        df[f"{args.text_column_name}_sentiment"] = sentiment_scores
        
        # Imputar los valores None/NaN en la nueva columna de sentimiento
        # Se imputarán con 0 para que el modelo pueda procesarlos como numéricos.
        df[f"{args.text_column_name}_sentiment"] = df[f"{args.text_column_name}_sentiment"].fillna(0) # Imputar con 0

    # Guardar los resultados
    os.makedirs(args.output_data, exist_ok=True)
    output_path = os.path.join(args.output_data, "processed_data_with_sentiment.csv")
    df.to_csv(output_path, index=False)
    print(f"Datos con sentimiento guardados en: {output_path}")

if __name__ == "__main__":
    main()
