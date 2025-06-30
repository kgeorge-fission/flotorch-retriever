from datetime import datetime
import json
from typing import Dict, List, Optional, Union
import uuid
from flotorch_core.embedding.embedding_registry import embedding_registry
from flotorch_core.embedding.guardrails.guardrails_embedding import GuardrailsEmbedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from flotorch_core.guardrails.guardrails import BedrockGuardrail
from flotorch_core.inferencer.guardrails.guardrails_inferencer import GuardRailsInferencer
from flotorch_core.inferencer.inferencer_provider_factory import InferencerProviderFactory
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.config.config import Config
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.rerank.rerank import BedrockReranker
from retriever.retriever import Retriever
from flotorch_core.storage.db.dynamodb import DynamoDB
from flotorch_core.storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage
from flotorch_core.storage.db.vector.open_search import OpenSearchClient
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.storage.db.vector.vector_storage_factory import VectorStorageFactory
from flotorch_core.storage.storage_provider_factory import StorageProviderFactory

from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class RetrieverProcessor(BaseFargateTaskProcessor):
    """
    Processor for retriever tasks in Fargate.
    """

    def process(self):
        logger.info("Starting retriever process.")
        try:
            # exp_config_data = self.input_data
            exp_config_data = {
                "experiment_id": "I52W6CPG",
                "execution_id": "HWQME",
                "bedrock_knowledge_base": True,
                "chunking_strategy": "",
                "chunk_overlap": 0,
                "chunk_size": 0,
                "directional_pricing": 0.36,
                "embedding_model": "",
                "embedding_service": "",
                "enable_context_guardrails": False,
                "enable_guardrails": False,
                "enable_prompt_guardrails": False,
                "enable_response_guardrails": False,
                "eval_cost_estimate": 0.27819306666666666,
                "eval_embedding_model": "amazon.titan-embed-image-v1",
                "eval_retrieval_model": "cohere.command-r-v1:0",
                "eval_service": "ragas",
                "gateway_api_key": "",
                "gateway_enabled": False,
                "gateway_url": "",
                "gt_data": "s3://flotorch-data-teseot/d5c529e3-b851-42ad-a89d-6de377b35982/gt_data/gt.json",
                "guardrail_id": "",
                "guardrail_name": "",
                "guardrail_version": "",
                "hierarchical_child_chunk_size": 0,
                "hierarchical_chunk_overlap_percentage": 0,
                "hierarchical_parent_chunk_size": 0,
                "indexing_algorithm": "",
                "indexing_cost_estimate": 0,
                "inferencing_cost_estimate": 0.048400000000000006,
                "is_opensearch": False,
                "kb_data": "SS317XLNBJ",
                "kb_name": "knowledge-base-Amazonbedrock",
                "knn_num": 3,
                "knowledge_base": True,
                "n_shot_prompts": 1,
                "n_shot_prompt_guide": {
                "examples": [
                    {
                    "answer": "Amazon Bedrock IDE within Amazon SageMaker Unified Studio is bound by the same SLAs as Amazon Bedrock. For more information, visit the Amazon Bedrock Service Level Agreement page.",
                    "question": "What are the Service Level Agreements (SLAs) for Amazon Bedrock IDE?"
                    },
                    {
                    "answer": "To facilitate a smooth onboarding experience with Amazon Bedrock IDE in Amazon SageMaker Unified Studio, you can find detailed documentation on the Amazon Bedrock IDE User Guide. If you have any additional questions or need further assistance, please don't hesitate to reach out to your AWS account team.",
                    "question": "What documentation and support resources are available for Amazon Bedrock IDE?"
                    },
                    {
                    "answer": "Amazon Bedrock IDE comes at no extra cost, and users only pay for the usage of the underlying resources that are required by the generative AI applications that they build. For example, customers will only pay for the associated model, Guardrail and Knowledge Base that they have used on their generative AI application. For more information, please visit the Amazon Bedrock pricing page.",
                    "question": "What are the pricing and billing models for using Amazon Bedrock IDE?"
                    }
                ],
                "system_prompt": "You are an expert AI researcher specializing in cloud computing and AWS services, particularly Amazon Bedrock. Your task is to extract information from the provided Amazon Bedrock PDF and generate precise, structured, and concise answers to user questions. Ensure each response is specific, directly relevant to the question, and clearly reflects the content of the document. If the information is not found in the provided context, respond only with 'The information is not available in the provided context.'",
                "user_prompt": "Based on the above retrieved context, answer the following question clearly and concisely, avoiding repetition:"
                },
                "region": "us-east-1",
                "rerank_model_id": "none",
                "retrieval_cost_estimate": 0.015798400000000004,
                "retrieval_model": "meta-textgeneration-llama-3-1-8b-instruct",
                "retrieval_service": "sagemaker",
                "temp_retrieval_llm": 0.3,
                "vector_dimension": 0
                }

            n_shot_prompt_guide_obj = get_n_shot_prompt_guide_obj(exp_config_data.get("execution_id"))
            exp_config_data["n_shot_prompt_guide_obj"] = n_shot_prompt_guide_obj

            logger.info(f"Into retriever processor. Processing event: {json.dumps(exp_config_data)}")

            gt_data = exp_config_data.get("gt_data")
            storage = StorageProviderFactory.create_storage_provider(gt_data)
            gt_data_path = storage.get_path(gt_data)
            json_reader = JSONReader(storage)

            if exp_config_data.get("knowledge_base", False) and not exp_config_data.get("bedrock_knowledge_base", False):
                embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
                embedding = embedding_class(
                    exp_config_data.get("embedding_model"), 
                    exp_config_data.get("aws_region"), 
                    int(exp_config_data.get("vector_dimension")))
                is_opensearch_required = True
            else:
                embedding = None
                is_opensearch_required = False
            
            if exp_config_data.get("enable_guardrails", False):
                base_guardrails = BedrockGuardrail(exp_config_data.get("guardrail_id", ""), exp_config_data.get("guardrail_version", 0), exp_config_data.get("aws_region", "us-east-1"))
            
            
            vector_storage = VectorStorageFactory.create_vector_storage(
                knowledge_base=exp_config_data.get("knowledge_base", False),
                use_bedrock_kb=exp_config_data.get("bedrock_knowledge_base", False),
                embedding=embedding,
                opensearch_host=config.get_opensearch_host() if is_opensearch_required else None,
                opensearch_port=config.get_opensearch_port() if is_opensearch_required else None,
                opensearch_username=config.get_opensearch_username() if is_opensearch_required else None,
                opensearch_password=config.get_opensearch_password() if is_opensearch_required else None,
                index_id=exp_config_data.get("index_id"),
                knowledge_base_id=exp_config_data.get("kb_data"),
                aws_region=exp_config_data.get("aws_region")
            )

            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_prompt_guardrails", False):
                vector_storage = GuardRailsVectorStorage(
                    vector_storage, 
                    base_guardrails,
                    exp_config_data.get("enable_prompt_guardrails", False),
                    exp_config_data.get("enable_context_guardrails", False)
                )

            reranker = BedrockReranker(exp_config_data.get("aws_region"), exp_config_data.get("rerank_model_id")) \
                if exp_config_data.get("rerank_model_id").lower() != "none" \
                else None
            
            inferencer = InferencerProviderFactory.create_inferencer_provider(
                exp_config_data.get("gateway_enabled", False),
                f'{exp_config_data.get("gateway_url", "")}/api/openai/v1',
                exp_config_data.get("gateway_api_key", ""),
                exp_config_data.get("retrieval_service"),
                exp_config_data.get("retrieval_model"), 
                exp_config_data.get("aws_region"), 
                config.get_sagemaker_arn_role(),
                int(exp_config_data.get("n_shot_prompts", 0)), 
                float(exp_config_data.get("temp_retrieval_llm", 0)), 
                exp_config_data.get("n_shot_prompt_guide_obj")
            )
            
            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_response_guardrails", False):
                inferencer = GuardRailsInferencer(inferencer, base_guardrails)

            retriever = Retriever(json_reader, embedding, vector_storage, inferencer, reranker)
            hierarchical = exp_config_data.get("chunking_strategy") == 'hierarchical'
            result, total_input_tokens, total_output_tokens = retriever.retrieve(
                gt_data_path, 
                "What is the patient's name?",
                int(exp_config_data.get("knn_num")), 
                hierarchical
            )

            # metrics
            batch_items = []
            for item in result:
                metrics = create_metrics(
                    exp_config_data.get("execution_id"),
                    exp_config_data.get("experiment_id"),
                    item.question,
                    item.gt_answer,
                    item.answer,
                    item.reference_contexts,
                    item.query_metadata,
                    item.answer_metadata,
                    item.guardrails_input_assessment,
                    item.guardrails_context_assessment,
                    item.guardrails_output_assessment,
                    exp_config_data.get("guardrail_id", ""),
                    item.guardrails_block_level,
                    item.guardrails_blocked
                )                

                batch_items.append(metrics)
                if len(batch_items) >= 25:
                    write_batch_to_metrics_dynamodb(batch_items)
                    batch_items = []

            if len(batch_items) > 0:
                write_batch_to_metrics_dynamodb(batch_items)

            experiment_id = exp_config_data.get("experiment_id")
            update_tokens(experiment_id, total_input_tokens,total_output_tokens)
            output = {"status": "success", "message": "Retriever completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during retriever process: {str(e)}")
            self.send_task_failure(str(e))

# get n shot prompt guide object
def get_n_shot_prompt_guide_obj(execution_id) -> Optional[Dict]:
    """
    Retrieves the n-shot prompt guide object from the dynamo db.
    """
    db = DynamoDB(config.get_execution_table_name())
    data = db.read({"id": execution_id})
    if data:
        n_shot_prompt_guide_obj = data[0].get("config", {}).get("n_shot_prompt_guide", None)
        return n_shot_prompt_guide_obj
    return None

# metrics: to be removed later
def create_metrics(
    execution_id,
    experiment_id,
    question: str,
    gt_answer: str,
    answer: str,
    reference_contexts: List[str],
    query_metadata: Dict[str, int],
    answer_metadata: Dict[str, int],
    guardrail_input_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_context_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_output_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_id: Optional[str] = None,
    guardrails_block_level: Optional[str] = '',
    guardrails_blocked: Optional[bool] = False
):
    metrics = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "execution_id": execution_id,
        "experiment_id": experiment_id,
        "question": question,
        "gt_answer": gt_answer,
        "generated_answer": answer,
        "reference_contexts": reference_contexts,
        "query_metadata": query_metadata,
        "answer_metadata": answer_metadata
    }

    if guardrails_blocked:
        metrics.update({
            "guardrail_input_assessment": guardrail_input_assessment,
            "guardrail_context_assessment": guardrail_context_assessment,
            "guardrail_output_assessment": guardrail_output_assessment,
            "guardrail_id": guardrail_id,
            "guardrail_blocked": guardrails_block_level
        })

    return metrics

def update_tokens(experiment_id, total_input_tokens, total_output_tokens):
    experiment_db = DynamoDB(config.get_experiment_table_name())
    token_fields = {
        "retrieval_input_tokens": total_input_tokens,
        "retrieval_output_tokens": total_output_tokens
    }
    if experiment_db.update({"id": experiment_id}, token_fields):
        logger.info(f"Updated tokens for experiment {experiment_id} successfully.")
    else:
        logger.error(f"Failed to update tokens for experiment {experiment_id}.")
    
    
def write_batch_to_metrics_dynamodb(batch_items: List[Dict]) -> None:
    """Write a batch of items to DynamoDB."""
    logger.info(f"Writing batch of {len(batch_items)} items to DynamoDB")
    db = DynamoDB(config.get_experiment_question_metrics_table())
    db.bulk_write(batch_items)
